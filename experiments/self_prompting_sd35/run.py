"""Orchestrate preparation, generation, OCR, reports, and collages."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .collage import render_case_collage, render_special_collage
from .data import (
    MODEL_KEYS,
    TARGET_KEYS,
    expand_jobs,
    prepare_samples,
    read_jsonl,
    resolve_path,
    validate_samples,
    write_jsonl,
)
from .report import evaluate, write_reports


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
VALID_STAGES = {"prepare", "textctrl", "self_prompting_sd35", "ocr", "report", "all"}


def selected_models(mode: str) -> tuple[str, ...]:
    if mode == "both":
        return MODEL_KEYS
    if mode in MODEL_KEYS:
        return (mode,)
    raise ValueError("mode must be one of: both, textctrl, self_prompting_sd35")


def experiment_output_path(value: str | Path) -> Path:
    """Resolve an output path and keep it inside the requested experiment tree."""

    output = resolve_path(PROJECT_ROOT, value)
    try:
        output.relative_to(EXPERIMENT_DIR)
    except ValueError as error:
        raise ValueError(
            f"output_dir must stay under {EXPERIMENT_DIR}; received {output}"
        ) from error
    return output


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


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def run_subprocess(
    command: Sequence[str | Path | int | float],
    *,
    cwd: Path = PROJECT_ROOT,
    repository: Path | None = None,
) -> None:
    executable = Path(str(command[0])).expanduser()
    if os.sep in str(command[0]) and not executable.is_file():
        raise FileNotFoundError(
            f"Python executable does not exist: {executable}. "
            "Set the corresponding *_PYTHON environment variable."
        )
    subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        env=command_environment(repository),
        check=True,
    )


def load_config(argv: Sequence[str]) -> tuple[DictConfig, Mapping[str, Any]]:
    config = OmegaConf.load(CONFIG_PATH)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(argv)))
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError("Experiment configuration must be a mapping")
    return config, resolved


def load_or_prepare_samples(
    config: Mapping[str, Any], output_dir: Path
) -> list[dict[str, Any]]:
    count = int(config["sample_count"])
    output_manifest = output_dir / "samples.jsonl"
    reuse_value = config.get("samples_path")
    if reuse_value not in (None, ""):
        source_manifest = require_file(
            resolve_path(PROJECT_ROOT, str(reuse_value)), "sample manifest"
        )
        samples = read_jsonl(source_manifest)
        validate_samples(samples, count)
        write_jsonl(output_manifest, samples)
        return samples
    if output_manifest.is_file() and not bool(config["overwrite"]):
        samples = read_jsonl(output_manifest)
        validate_samples(samples, count)
        return samples
    return prepare_samples(config, PROJECT_ROOT, output_dir)


def run_textctrl(config: Mapping[str, Any], manifest: Path) -> None:
    section = config["textctrl"]
    repository = resolve_path(PROJECT_ROOT, str(section["repository_dir"]))
    checkpoint = require_file(
        resolve_path(PROJECT_ROOT, str(section["checkpoint_path"])),
        "TextCtrl checkpoint",
    )
    require_file(repository / "configs" / "inference.yaml", "TextCtrl inference config")
    command: list[str | Path | int | float] = [
        section["python_executable"],
        EXPERIMENT_DIR / "textctrl_worker.py",
        "--repository",
        repository,
        "--checkpoint",
        checkpoint,
        "--manifest",
        manifest,
        "--starting-layer",
        section["starting_layer"],
        "--num-inference-steps",
        section["num_inference_steps"],
        "--guidance-scale",
        section["guidance_scale"],
    ]
    if bool(config["overwrite"]):
        command.append("--overwrite")
    run_subprocess(command, cwd=repository, repository=repository)


def run_self_prompting_sd35(config: Mapping[str, Any], manifest: Path) -> None:
    section = config["self_prompting_sd35"]
    checkpoint = resolve_path(PROJECT_ROOT, str(section["checkpoint_path"]))
    require_file(
        checkpoint / "pytorch_lora_weights.safetensors",
        "self-prompting SD3.5 LoRA weights",
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
        "--font-path",
        resolve_path(PROJECT_ROOT, str(section["font_path"])),
    ]
    if bool(config["overwrite"]):
        command.append("--overwrite")
    run_subprocess(command)


def run_ocr(config: Mapping[str, Any], manifest: Path, predictions: Path) -> None:
    section = config["ocr"]
    repository = resolve_path(PROJECT_ROOT, str(section["repository_dir"]))
    checkpoint = require_file(
        resolve_path(PROJECT_ROOT, str(section["checkpoint_path"])), "OCR checkpoint"
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


def run_report(
    config: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    predictions_path: Path,
    output_dir: Path,
    models: Sequence[str],
) -> list[Path]:
    require_file(predictions_path, "OCR prediction manifest")
    prediction_rows = read_jsonl(predictions_path)
    predictions = {int(row["index"]): row for row in prediction_rows}
    if len(predictions) != len(jobs):
        raise RuntimeError(
            f"OCR results are incomplete: expected {len(jobs)}, found {len(predictions)}"
        )
    rows = evaluate(
        jobs,
        predictions,
        case_sensitive=bool(config["metrics"]["case_sensitive"]),
    )
    outputs = write_reports(rows, output_dir)
    if tuple(models) == MODEL_KEYS:
        outputs.append(
            render_case_collage(
                samples,
                rows,
                output_dir / "capital_lowercase_collage.png",
                config["collage"],
            )
        )
        outputs.append(
            render_special_collage(
                samples,
                rows,
                output_dir / "special_character_collage.png",
                config["collage"],
            )
        )
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    config_node, config = load_config(raw_arguments)
    stage = str(config["stage"])
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of: {', '.join(sorted(VALID_STAGES))}")
    models = selected_models(str(config["mode"]))
    if stage in MODEL_KEYS and stage not in models:
        raise ValueError(f"stage={stage} is incompatible with mode={config['mode']}")

    output_dir = experiment_output_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_or_prepare_samples(config, output_dir)
    jobs = expand_jobs(samples, config, output_dir, models=models)
    expected_jobs = len(samples) * len(TARGET_KEYS) * len(models)
    if len(jobs) != expected_jobs:
        raise RuntimeError(f"Expected {expected_jobs} jobs, constructed {len(jobs)}")
    manifest = output_dir / "jobs.jsonl"
    write_jsonl(manifest, jobs)
    OmegaConf.save(config_node, output_dir / "config.yaml")

    if stage == "prepare":
        print(
            json.dumps(
                {
                    "mode": config["mode"],
                    "samples": len(samples),
                    "jobs": len(jobs),
                    "output_dir": str(output_dir),
                },
                indent=2,
            )
        )
        return

    if "textctrl" in models and stage in ("textctrl", "all"):
        run_textctrl(config, manifest)
    if "self_prompting_sd35" in models and stage in ("self_prompting_sd35", "all"):
        run_self_prompting_sd35(config, manifest)

    predictions_path = output_dir / "ocr_predictions.jsonl"
    if stage in ("ocr", "all"):
        run_ocr(config, manifest, predictions_path)
    report_outputs: list[Path] = []
    if stage in ("report", "all"):
        report_outputs = run_report(
            config,
            samples,
            jobs,
            predictions_path,
            output_dir,
            models,
        )

    print(
        json.dumps(
            {
                "mode": config["mode"],
                "stage": stage,
                "samples": len(samples),
                "jobs": len(jobs),
                "reports": [str(path) for path in report_outputs],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

