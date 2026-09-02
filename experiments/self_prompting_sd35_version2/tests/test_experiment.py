from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from experiments.self_prompting_sd35_version2 import ocr_worker, sd35_worker
from experiments.self_prompting_sd35_version2.collage import (
    load_previous_sd35_rows,
    render_case_collage,
    render_special_collage,
)
from experiments.self_prompting_sd35_version2.data import (
    MODEL_KEY,
    TARGET_KEYS,
    expand_jobs,
    load_target_rows,
    prepare_samples,
    read_jsonl,
)
from experiments.self_prompting_sd35_version2.metrics import mean_std, text_metrics
from experiments.self_prompting_sd35_version2.report import (
    MODEL_DISPLAY,
    evaluate,
    write_reports,
)
from experiments.self_prompting_sd35_version2.provenance import sha256_file
from experiments.self_prompting_sd35_version2.run import (
    EXPERIMENT_DIR,
    experiment_output_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREVIOUS_RESULTS = (
    PROJECT_ROOT / "experiments" / "self_prompting_sd35_version1" / "results"
)
VERSION2_CHECKPOINT = (
    PROJECT_ROOT
    / "CODEX"
    / "self_prompting_sd35"
    / "checkpoints"
    / "version2"
    / "checkpoint-030000"
)


def test_metrics_are_case_sensitive_and_use_population_std() -> None:
    assert text_metrics("ABCDE", "abcde") == {"ACC": 0.0, "NED": 0.0, "CER": 1.0}
    assert text_metrics("ABCDE", "abcde", case_sensitive=False) == {
        "ACC": 1.0,
        "NED": 1.0,
        "CER": 0.0,
    }
    assert text_metrics("abcde", "abXde") == {"ACC": 0.0, "NED": 0.8, "CER": 0.2}
    assert mean_std([0.0, 1.0]) == (0.5, 0.5)


def test_real_v1_provenance_reconstructs_exact_100_by_8_grid(tmp_path: Path) -> None:
    samples = prepare_samples(PREVIOUS_RESULTS / "samples.jsonl", tmp_path)
    targets = load_target_rows(
        PREVIOUS_RESULTS / "detailed_results.csv", samples
    )
    jobs = expand_jobs(samples, targets, tmp_path)

    assert len(samples) == 100
    assert len(targets) == 800
    assert len(jobs) == 800
    assert {job["model"] for job in jobs} == {MODEL_KEY}
    assert {
        (int(job["sample_index"]), str(job["target_key"])) for job in jobs
    } == {(sample_index, key) for sample_index in range(100) for key in TARGET_KEYS}

    prior_samples = read_jsonl(PREVIOUS_RESULTS / "samples.jsonl")
    for prior, current in zip(prior_samples, samples, strict=True):
        actual_prior_input = PREVIOUS_RESULTS / "inputs" / Path(prior["input_path"]).name
        assert Path(current["input_path"]).read_bytes() == actual_prior_input.read_bytes()
        assert current["filename"] == prior["filename"]
        assert current["noise_seed"] == prior["noise_seed"]
        assert current["noise_standard_deviation"] == prior["noise_standard_deviation"]

    with (PREVIOUS_RESULTS / "detailed_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        prior_rows = list(csv.DictReader(handle))
    exact_targets = {
        (row["filename"], row["target_key"], row["target_text"])
        for row in prior_rows
    }
    current_targets = {
        (job["filename"], job["target_key"], job["target_text"]) for job in jobs
    }
    assert current_targets == exact_targets


def _worker_options(checkpoint: Path) -> argparse.Namespace:
    return argparse.Namespace(
        resolution=512,
        steps=28,
        max_sequence_length=256,
        base_model="base",
        dtype="bfloat16",
        checkpoint=checkpoint,
        font_path=None,
    )


def test_checkpoint_validation_requires_both_v2_artifacts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / sd35_worker.LORA_WEIGHT_NAME).write_bytes(b"lora")
    with pytest.raises(sd35_worker.WorkerError, match="input_projection"):
        sd35_worker.validate_options(_worker_options(checkpoint))
    (checkpoint / sd35_worker.INPUT_PROJECTION_NAME).write_bytes(b"projection")
    options = _worker_options(checkpoint)
    sd35_worker.validate_options(options)
    assert options.checkpoint == checkpoint.resolve()


def test_actual_checkpoint_contains_both_required_artifacts() -> None:
    options = _worker_options(VERSION2_CHECKPOINT)
    sd35_worker.validate_options(options)


def test_loader_uses_wrapper_loader_after_model_construction(tmp_path: Path) -> None:
    events: list[object] = []

    class FakePipe:
        @classmethod
        def from_pretrained(cls, name, dtype):
            events.append(("pipeline", name, dtype))
            return cls()

        def to(self, device):
            events.append(("pipeline.to", device))
            return self

        def load_lora_weights(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("Pipeline loader must not load a v2 checkpoint")

    class FakeModel:
        def __init__(self, pipe):
            events.append(("model", pipe))

        def to(self, device):
            events.append(("model.to", device))
            return self

        def load_lora_weights(self, checkpoint):
            events.append(("model.load", checkpoint))

        def eval(self):
            events.append("model.eval")
            return self

    checkpoint = tmp_path / "checkpoint"
    pipe, model = sd35_worker._load_pipeline_and_adapter(
        torch=object(),
        pipeline_class=FakePipe,
        model_class=FakeModel,
        base_model="base-model",
        checkpoint=checkpoint,
        torch_dtype="bf16",
        device="cuda",
    )
    assert isinstance(pipe, FakePipe)
    assert isinstance(model, FakeModel)
    assert events[-2:] == [("model.load", checkpoint), "model.eval"]


def _synthetic_evaluation(tmp_path: Path, sample_count: int = 5):
    samples = []
    jobs = []
    predictions = {}
    for sample_index in range(sample_count):
        source = tmp_path / "inputs" / f"{sample_index:04d}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 24), (100, 120, 140)).save(source)
        sample = {
            "sample_index": sample_index,
            "filename": f"{sample_index:05d}.png",
            "source_text": "abcde",
            "input_path": str(source),
        }
        samples.append(sample)
        for target_key in TARGET_KEYS:
            output = tmp_path / "generated" / target_key / f"{sample_index:04d}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (48, 24), "white").save(output)
            job = {
                "index": len(jobs),
                "sample_index": sample_index,
                "filename": sample["filename"],
                "model": MODEL_KEY,
                "target_key": target_key,
                "source_text": "abcde",
                "target_text": "ABCDE" if target_key == "case_upper" else "abcde",
                "noise_standard_deviation": 3.0,
                "noise_seed": sample_index,
                "generation_seed": 100 + len(jobs),
                "source_path": str(source),
                "input_path": str(source),
                "mask_path": str(source),
                "output_path": str(output),
            }
            jobs.append(job)
            predictions[job["index"]] = {
                "index": job["index"],
                "ocr_predicted_text": job["target_text"],
                "output_path": str(output),
            }
    return samples, jobs, evaluate(jobs, predictions)


def test_reports_preserve_baselines_and_append_v2_rows(tmp_path: Path) -> None:
    _, _, rows = _synthetic_evaluation(tmp_path)
    case_source = PREVIOUS_RESULTS / "capital_lowercase_summary.csv"
    special_source = PREVIOUS_RESULTS / "special_character_summary.csv"
    with case_source.open(newline="", encoding="utf-8") as handle:
        original_case = list(csv.reader(handle))
    with special_source.open(newline="", encoding="utf-8") as handle:
        original_special = list(csv.reader(handle))

    outputs = write_reports(
        rows,
        case_source,
        special_source,
        tmp_path / "reports",
        expected_samples=5,
    )
    assert len(outputs) == 3
    assert all(path.is_file() for path in outputs)
    with outputs[1].open(newline="", encoding="utf-8") as handle:
        current_case = list(csv.reader(handle))
    with outputs[2].open(newline="", encoding="utf-8") as handle:
        current_special = list(csv.reader(handle))

    for baseline_row in (original_case[2], original_case[3], original_case[7], original_case[8]):
        assert baseline_row in current_case
    assert sum(bool(row) and row[0] == MODEL_DISPLAY for row in current_case) == 2
    assert current_special[1:3] == original_special[1:3]
    assert current_special[-1][0] == MODEL_DISPLAY
    assert len(current_special[-1]) == 19


def test_collages_have_literal_requested_dimensions(tmp_path: Path) -> None:
    samples, _, rows = _synthetic_evaluation(tmp_path)
    previous_rows = []
    for sample_index in range(5):
        output = tmp_path / "v1" / f"{sample_index:04d}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 24), "gray").save(output)
        previous_rows.append(
            {
                "sample_index": sample_index,
                "target_key": "case_upper",
                "target_text": "ABCDE",
                "ACC": "1.0",
                "NED": "1.0",
                "CER": "0.0",
                "output_path": str(output),
            }
        )
    config = {
        "font_path": None,
        "font_size": 10,
        "cell_width": 100,
        "image_height": 40,
        "caption_height": 50,
    }
    case_path = render_case_collage(
        samples, rows, previous_rows, tmp_path / "case.png", config
    )
    special_path = render_special_collage(
        samples, rows, tmp_path / "special.png", config
    )
    with Image.open(case_path) as image:
        assert image.size == (3 * 100, 5 * 90)
    with Image.open(special_path) as image:
        assert image.size == (7 * 100, 1 * 90)


def test_v1_collage_loader_repairs_old_generated_paths() -> None:
    rows = load_previous_sd35_rows(
        PREVIOUS_RESULTS / "detailed_results.csv", PREVIOUS_RESULTS
    )
    assert len(rows) == 800
    assert all(Path(row["output_path"]).is_file() for row in rows)


def test_output_directory_is_confined_to_v2_experiment() -> None:
    expected = EXPERIMENT_DIR / "results"
    assert experiment_output_path(
        "experiments/self_prompting_sd35_version2/results"
    ) == expected
    with pytest.raises(ValueError, match="must stay under"):
        experiment_output_path("experiments/self_prompting_sd35_version1/results")
    with pytest.raises(ValueError, match="must stay under"):
        experiment_output_path("/tmp/version2-results")


def test_ocr_resume_finishes_without_importing_gpu_stack(tmp_path: Path) -> None:
    repository = tmp_path / "TextCtrl"
    (repository / "configs").mkdir(parents=True)
    (repository / "configs" / "inference.yaml").write_text("config", encoding="utf-8")
    package = repository / "src" / "module" / "abinet"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package", encoding="utf-8")
    checkpoint = repository / "ocr_model.pth"
    checkpoint.write_bytes(b"checkpoint")
    image = tmp_path / "generated.png"
    Image.new("RGB", (20, 10), "white").save(image)
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "index": 0,
                "model": MODEL_KEY,
                "output_path": str(image),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "ocr_predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "index": 0,
                "model": MODEL_KEY,
                "ocr_predicted_text": "abcde",
                "output_path": str(image),
                "image_sha256": sha256_file(image),
                "ocr_checkpoint_sha256": sha256_file(checkpoint),
                "ocr_config_sha256": sha256_file(repository / "configs" / "inference.yaml"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert ocr_worker.main(
        [
            "--repository",
            str(repository),
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
        ]
    ) == 0
