from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from experiments.self_prompting_sd35 import ocr_worker
from experiments.self_prompting_sd35 import run as experiment_run

from experiments.self_prompting_sd35.collage import (
    render_case_collage,
    render_special_collage,
)
from experiments.self_prompting_sd35.data import (
    MODEL_KEYS,
    SPECIAL_PATTERNS,
    TARGET_KEYS,
    expand_jobs,
    prepare_samples,
)
from experiments.self_prompting_sd35.metrics import mean_std, text_metrics
from experiments.self_prompting_sd35.report import evaluate, write_reports
from experiments.self_prompting_sd35.run import EXPERIMENT_DIR, experiment_output_path


def dataset_config(shard: Path, sample_count: int = 1) -> dict:
    return {
        "sample_count": sample_count,
        "sample_seed": 11,
        "target_seed": 22,
        "generation_seed": 33,
        "dataset": {
            "roots": [str(shard)],
            "source_labels": "i_s.txt",
            "source_dir": "i_s",
            "mask_dir": "mask_s",
        },
        "input_noise": {
            "seed": 44,
            "minimum_standard_deviation": 3.0,
            "maximum_standard_deviation": 3.0,
        },
        "targets": {
            "letters": "abcXYZ",
            "special_characters": "!?",
        },
    }


def create_shard(root: Path, count: int = 1) -> Path:
    shard = root / "shard"
    (shard / "i_s").mkdir(parents=True)
    (shard / "mask_s").mkdir()
    labels = []
    for index in range(count):
        name = f"{index:05d}.png"
        Image.new("RGB", (40 + index, 20), (80, 120, 160)).save(shard / "i_s" / name)
        Image.new("L", (40 + index, 20), 255).save(shard / "mask_s" / name)
        labels.append(f"{name} abcde\n")
    (shard / "i_s.txt").write_text("".join(labels), encoding="utf-8")
    return shard


def test_metrics_are_case_sensitive_and_use_population_std() -> None:
    assert text_metrics("ABCDE", "abcde") == {"ACC": 0.0, "NED": 0.0, "CER": 1.0}
    assert text_metrics("ABCDE", "abcde", case_sensitive=False) == {
        "ACC": 1.0,
        "NED": 1.0,
        "CER": 0.0,
    }
    assert text_metrics("abcde", "abXde") == {"ACC": 0.0, "NED": 0.8, "CER": 0.2}
    assert mean_std([0.0, 1.0]) == (0.5, 0.5)


def test_target_families_noise_and_model_pairing_are_deterministic(tmp_path: Path) -> None:
    shard = create_shard(tmp_path)
    config = dataset_config(shard)
    first = prepare_samples(config, tmp_path, tmp_path / "first")
    second = prepare_samples(config, tmp_path, tmp_path / "second")
    assert first[0]["case_upper"] == first[0]["case_lower"].upper()
    assert {
        key: first[0][key] for key in TARGET_KEYS
    } == {key: second[0][key] for key in TARGET_KEYS}
    assert Path(first[0]["input_path"]).read_bytes() == Path(
        second[0]["input_path"]
    ).read_bytes()

    letters, specials = set(config["targets"]["letters"]), set(
        config["targets"]["special_characters"]
    )
    for target_index, (letter_count, special_count) in enumerate(SPECIAL_PATTERNS, 1):
        target = first[0][f"special_{target_index}"]
        assert sum(character in letters for character in target) == letter_count
        assert sum(character in specials for character in target) == special_count

    jobs = expand_jobs(first, config, tmp_path / "jobs")
    assert len(jobs) == len(MODEL_KEYS) * len(TARGET_KEYS)
    paired = {
        (job["model"], job["target_key"]): (
            job["target_text"],
            job["generation_seed"],
        )
        for job in jobs
    }
    for key in TARGET_KEYS:
        assert paired[("textctrl", key)] == paired[("self_prompting_sd35", key)]


def test_reports_and_collages_have_requested_shapes(tmp_path: Path) -> None:
    shard = create_shard(tmp_path, count=5)
    config = dataset_config(shard, sample_count=5)
    samples = prepare_samples(config, tmp_path, tmp_path / "prepared")
    jobs = expand_jobs(samples, config, tmp_path / "output")
    predictions = {}
    for job in jobs:
        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 24), "white").save(output_path)
        predictions[int(job["index"])] = {
            "index": int(job["index"]),
            "ocr_predicted_text": job["target_text"],
            "output_path": str(output_path),
        }
    rows = evaluate(jobs, predictions)
    report_paths = write_reports(rows, tmp_path / "reports")
    assert all(path.is_file() for path in report_paths)

    predictions_path = tmp_path / "ocr_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in predictions.values()),
        encoding="utf-8",
    )
    sd35_csv = experiment_run.write_sd35_ocr_results(
        {"metrics": {"case_sensitive": True}},
        jobs,
        predictions_path,
        tmp_path / "reports",
    )
    with sd35_csv.open(newline="", encoding="utf-8") as handle:
        sd35_rows = list(csv.DictReader(handle))
    assert len(sd35_rows) == 5 * len(TARGET_KEYS)
    assert {row["model"] for row in sd35_rows} == {"self_prompting_sd35"}

    with (tmp_path / "reports" / "special_character_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        special_rows = list(csv.reader(handle))
    assert len(special_rows[0]) == 19  # model + 18 metric cells
    assert [row[0] for row in special_rows[1:]] == [
        "TextCtrl",
        "Self Prompting SD3.5",
    ]

    collage_config = {
        "font_path": None,
        "font_size": 10,
        "cell_width": 100,
        "image_height": 40,
        "caption_height": 50,
    }
    case_path = render_case_collage(
        samples, rows, tmp_path / "case.png", collage_config
    )
    special_path = render_special_collage(
        samples, rows, tmp_path / "special.png", collage_config
    )
    with Image.open(case_path) as case_image:
        assert case_image.size == (5 * 100, 5 * 90)
    with Image.open(special_path) as special_image:
        assert special_image.size == (7 * 100, 2 * 90)


def test_output_directory_is_confined_to_experiment() -> None:
    assert experiment_output_path("experiments/self_prompting_sd35/results") == (
        EXPERIMENT_DIR / "results"
    )
    with pytest.raises(ValueError, match="must stay under"):
        experiment_output_path("/tmp/self_prompting_sd35-results")



def test_all_stage_runs_sd35_csv_before_textctrl(monkeypatch, tmp_path: Path) -> None:
    events = []
    config = {"overwrite": True}

    monkeypatch.setattr(
        experiment_run,
        "run_self_prompting_sd35",
        lambda *_: events.append("sd35_generate"),
    )

    def fake_ocr(*_, model=None, overwrite=None):
        events.append(f"ocr:{model}:{overwrite}")

    monkeypatch.setattr(experiment_run, "run_ocr", fake_ocr)

    sd35_csv = tmp_path / "self_prompting_sd35_ocr_results.csv"

    def fake_sd35_csv(*_):
        events.append("sd35_csv")
        return sd35_csv

    monkeypatch.setattr(experiment_run, "write_sd35_ocr_results", fake_sd35_csv)
    monkeypatch.setattr(
        experiment_run, "run_textctrl", lambda *_: events.append("textctrl_generate")
    )

    final_report = tmp_path / "detailed_results.csv"

    def fake_report(*_):
        events.append("combined_report")
        return [final_report]

    monkeypatch.setattr(experiment_run, "run_report", fake_report)
    outputs = experiment_run.run_all_stages(
        config, [], [], tmp_path / "jobs.jsonl", tmp_path / "ocr.jsonl",
        tmp_path, MODEL_KEYS,
    )

    assert events == [
        "sd35_generate",
        "ocr:self_prompting_sd35:True",
        "sd35_csv",
        "textctrl_generate",
        "ocr:textctrl:False",
        "combined_report",
    ]
    assert outputs == [sd35_csv, final_report]


def test_filtered_ocr_does_not_require_other_model_outputs(tmp_path: Path) -> None:
    repository = tmp_path / "TextCtrl"
    (repository / "configs").mkdir(parents=True)
    (repository / "configs" / "inference.yaml").write_text("config", encoding="utf-8")
    package = repository / "src" / "module" / "abinet"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package", encoding="utf-8")
    checkpoint = repository / "ocr_model.pth"
    checkpoint.write_bytes(b"checkpoint")

    sd35_output = tmp_path / "sd35.png"
    Image.new("RGB", (20, 10), "white").save(sd35_output)
    textctrl_output = tmp_path / "not-generated-yet.png"
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"index": 0, "model": "textctrl", "output_path": str(textctrl_output)}),
                json.dumps({"index": 1, "model": "self_prompting_sd35", "output_path": str(sd35_output)}),
            ]
        ) + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "ocr_predictions.jsonl"
    predictions.write_text(
        json.dumps({"index": 1, "ocr_predicted_text": "abcde", "output_path": str(sd35_output)}) + "\n",
        encoding="utf-8",
    )

    ocr_worker.main(
        [
            "--repository", str(repository),
            "--checkpoint", str(checkpoint),
            "--manifest", str(manifest),
            "--predictions", str(predictions),
            "--model", "self_prompting_sd35",
        ]
    )
