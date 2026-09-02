"""Prepare, generate, OCR, and report the version-2 evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .collage import (
    load_previous_sd35_rows,
    render_case_collage,
    render_special_collage,
)
from .data import (
    MODEL_KEY,
    TARGET_KEYS,
    expand_jobs,
    load_target_rows,
    prepare_samples,
    read_jsonl,
    write_jsonl,
)
from .report import evaluate, write_reports


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
RESULTS_DIR = EXPERIMENT_DIR / "results"
VALID_STAGES = {"prepare", "generate", "ocr", "report", "all"}


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def experiment_output_path(value: str | Path) -> Path:
    """Resolve and confine outputs to this experiment's results tree."""

    output = resolve_path(value)
    try:
        output.relative_to(RESULTS_DIR)
    except ValueError as exc:
        raise ValueError(
            f"output_dir must stay under {RESULTS_DIR}; received {output}"
        ) from exc
    return output


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{description.capitalize()} is empty: {path}")
    return path


def command_environment(repository: Path | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    paths = [str(PROJECT_ROOT)]
    if repository is not None:
        paths.append(str(repository))
    existing = environment.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def run_subprocess(
    command: Sequence[str | Path | int | float],
    *,
    cwd: Path = PROJECT_ROOT,
    repository: Path | None = None,
) -> None:
    executable = Path(str(command[0])).expanduser()
    if os.sep in str(command[0]) and not executable.is_file():
        raise FileNotFoundError(
            f"Python executable does not exist: {executable}. Override its *_PYTHON "
            "environment variable or the matching config value."
        )
    subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        env=command_environment(repository),
        check=True,
    )


def load_config(argv: Sequence[str]) -> tuple[DictConfig, Mapping[str, Any]]:
    config = OmegaConf.merge(
        OmegaConf.load(CONFIG_PATH), OmegaConf.from_dotlist(list(argv))
    )
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("Experiment configuration must resolve to a mapping")
    return config, resolved


def prepare_reference_jobs(
    config: Mapping[str, Any], output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize the exact v1 noisy samples and target pairings locally."""

    expected_count = int(config["sample_count"])
    previous_samples = require_file(
        resolve_path(str(config["previous_samples_path"])),
        "version-1 sample manifest",
    )
    previous_detailed = require_file(
        resolve_path(str(config["previous_detailed_results_path"])),
        "version-1 detailed results",
    )
    samples = prepare_samples(
        previous_samples,
        output_dir,
        expected_count=expected_count,
        overwrite=bool(config["overwrite"]),
    )
    target_rows = load_target_rows(
        previous_detailed,
        samples,
        expected_count=expected_count,
    )
    jobs = expand_jobs(samples, target_rows, output_dir, model=MODEL_KEY)
    expected_jobs = expected_count * len(TARGET_KEYS)
    if len(jobs) != expected_jobs:
        raise RuntimeError(f"Expected {expected_jobs} v2 jobs, constructed {len(jobs)}")
    write_jsonl(output_dir / "samples.jsonl", samples)
    write_jsonl(output_dir / "jobs.jsonl", jobs)
    return samples, jobs


def run_generation(config: Mapping[str, Any], manifest: Path) -> None:
    section = config["self_prompting_sd35"]
    checkpoint = resolve_path(str(section["checkpoint_path"]))
    require_file(
        checkpoint / "pytorch_lora_weights.safetensors",
        "version-2 attention LoRA weights",
    )
    require_file(
        checkpoint / "input_projection.safetensors",
        "version-2 learned 65-channel input projection",
    )
    command: list[str | Path | int | float] = [
        section["python_executable"],
        EXPERIMENT_DIR / "sd35_worker.py",
        "--manifest",
        manifest,
        "--checkpoint",
        checkpoint,
        "--base-model",
        section["base_model"],
        "--resolution",
        section["resolution"],
        "--steps",
        section["num_inference_steps"],
        "--max-sequence-length",
        section["max_sequence_length"],
        "--dtype",
        section["dtype"],
    ]
    revision = section.get("revision")
    if revision not in (None, ""):
        command.extend(["--revision", str(revision)])
    font_path = section.get("font_path")
    if font_path not in (None, ""):
        command.extend(["--font-path", resolve_path(str(font_path))])
    if bool(config["overwrite"]):
        command.append("--overwrite")
    run_subprocess(command)


def run_ocr(
    config: Mapping[str, Any], manifest: Path, predictions: Path
) -> None:
    section = config["ocr"]
    repository = resolve_path(str(section["repository_dir"]))
    require_file(repository / "configs" / "inference.yaml", "TextCtrl inference config")
    checkpoint = require_file(
        resolve_path(str(section["checkpoint_path"])), "ABINet OCR checkpoint"
    )
    command: list[str | Path] = [
        section["python_executable"],
        EXPERIMENT_DIR / "ocr_worker.py",
        "--repository",
        repository,
        "--checkpoint",
        checkpoint,
        "--manifest",
        manifest,
        "--predictions",
        predictions,
    ]
    if bool(config["overwrite"]):
        command.append("--overwrite")
    run_subprocess(command, cwd=repository, repository=repository)


def _prediction_map(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(require_file(path, "OCR prediction manifest"))
    predictions: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "index" not in row:
            raise ValueError("OCR prediction is missing index")
        index = int(row["index"])
        if index in predictions:
            raise ValueError(f"Duplicate OCR prediction index {index}")
        predictions[index] = row
    return predictions


def run_report(
    config: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    predictions_path: Path,
    output_dir: Path,
) -> list[Path]:
    predictions = _prediction_map(predictions_path)
    if len(predictions) != len(jobs):
        raise RuntimeError(
            f"OCR results are incomplete: expected {len(jobs)}, found {len(predictions)}"
        )
    rows = evaluate(
        jobs,
        predictions,
        case_sensitive=bool(config["metrics"]["case_sensitive"]),
    )
    previous_case = require_file(
        resolve_path(str(config["previous_case_summary_path"])),
        "version-1 capitalization summary",
    )
    previous_special = require_file(
        resolve_path(str(config["previous_special_summary_path"])),
        "version-1 special-character summary",
    )
    outputs = write_reports(
        rows,
        previous_case,
        previous_special,
        output_dir,
        expected_samples=int(config["sample_count"]),
    )

    previous_detailed = require_file(
        resolve_path(str(config["previous_detailed_results_path"])),
        "version-1 detailed results",
    )
    previous_results_dir = resolve_path(str(config["previous_results_dir"]))
    version1_rows = load_previous_sd35_rows(previous_detailed, previous_results_dir)
    outputs.extend(
        [
            render_case_collage(
                samples,
                rows,
                version1_rows,
                output_dir / "capital_lowercase_collage.png",
                config["collage"],
            ),
            render_special_collage(
                samples,
                rows,
                output_dir / "special_character_collage.png",
                config["collage"],
            ),
        ]
    )
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    _, config = load_config(raw_arguments)
    stage = str(config["stage"])
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of: {', '.join(sorted(VALID_STAGES))}")
    if config["metrics"].get("case_sensitive") is not True:
        raise ValueError(
            "metrics.case_sensitive must remain true so v2 metrics are comparable "
            "with the copied case-sensitive v1 baselines"
        )
    output_dir = experiment_output_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, jobs = prepare_reference_jobs(config, output_dir)
    manifest = output_dir / "jobs.jsonl"
    predictions = output_dir / "ocr_predictions.jsonl"
    OmegaConf.save(OmegaConf.create(config), output_dir / "config.yaml")

    reports: list[Path] = []
    if stage == "prepare":
        pass
    elif stage == "generate":
        run_generation(config, manifest)
    elif stage == "ocr":
        run_ocr(config, manifest, predictions)
    elif stage == "report":
        reports = run_report(config, samples, jobs, predictions, output_dir)
    elif stage == "all":
        print("Generating Self-Prompting SD3.5 version-2 edits...", flush=True)
        run_generation(config, manifest)
        print("Recognizing version-2 edits with ABINet...", flush=True)
        run_ocr(config, manifest, predictions)
        print("Writing detailed metrics, augmented summaries, and collages...", flush=True)
        reports = run_report(config, samples, jobs, predictions, output_dir)

    print(
        json.dumps(
            {
                "stage": stage,
                "samples": len(samples),
                "jobs": len(jobs),
                "model": MODEL_KEY,
                "reports": [str(path) for path in reports],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
