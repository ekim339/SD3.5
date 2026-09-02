"""Run TextCtrl's released ABINet over every generated version-2 image."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from .ocr_provenance import (
        OCRProvenanceError,
        validate_completed as validate_completed_provenance,
    )
    from .provenance import sha256_file
except ImportError:  # Direct execution: the worker directory is on sys.path.
    from ocr_provenance import (  # type: ignore[no-redef]
        OCRProvenanceError,
        validate_completed as validate_completed_provenance,
    )
    from provenance import sha256_file  # type: ignore[no-redef]


MODEL_KEY = "self_prompting_sd35_version2"
REQUIRED_JOB_FIELDS = {"index", "model", "output_path"}


class OCRError(RuntimeError):
    """An actionable OCR configuration or manifest error."""


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise OCRError(f"Missing {description}: {path}")
    if path.stat().st_size == 0:
        raise OCRError(f"{description.capitalize()} is empty: {path}")
    return path


def _read_json_objects(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OCRError(
                f"Invalid JSON in {description} {path} at line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise OCRError(
                f"{description.capitalize()} {path} line {line_number} must be an object"
            )
        rows.append(value)
    return rows


def read_manifest(path: Path) -> list[dict[str, Any]]:
    """Read and strictly validate a v2 job manifest."""

    _require_file(path, "job manifest")
    jobs: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_outputs: set[Path] = set()
    for line_number, value in enumerate(_read_json_objects(path, "manifest"), 1):
        missing = sorted(REQUIRED_JOB_FIELDS.difference(value))
        if missing:
            raise OCRError(
                f"Manifest row {line_number} is missing fields: {', '.join(missing)}"
            )
        if value["model"] != MODEL_KEY:
            raise OCRError(
                f"Manifest row {line_number} has model={value['model']!r}; "
                f"expected {MODEL_KEY!r}"
            )
        try:
            index = int(value["index"])
        except (TypeError, ValueError) as exc:
            raise OCRError(f"Manifest row {line_number} has a non-integer index") from exc
        if index in seen_indices:
            raise OCRError(f"Duplicate manifest index {index}")
        output_path = _resolve(str(value["output_path"]), path.parent)
        if output_path in seen_outputs:
            raise OCRError(f"Duplicate manifest output path: {output_path}")
        seen_indices.add(index)
        seen_outputs.add(output_path)
        row = dict(value)
        row["index"] = index
        row["_output_path"] = output_path
        jobs.append(row)
    if not jobs:
        raise OCRError(f"Job manifest is empty: {path}")
    return jobs


def read_predictions(path: Path) -> dict[int, dict[str, Any]]:
    """Read completed OCR rows for resumable execution."""

    if not path.exists():
        return {}
    if not path.is_file():
        raise OCRError(f"Predictions path is not a file: {path}")
    completed: dict[int, dict[str, Any]] = {}
    for line_number, value in enumerate(_read_json_objects(path, "predictions"), 1):
        required = {
            "index",
            "model",
            "ocr_predicted_text",
            "output_path",
            "image_sha256",
            "ocr_checkpoint_sha256",
            "ocr_config_sha256",
        }
        missing = required.difference(value)
        if missing:
            raise OCRError(
                f"Predictions row {line_number} is missing fields: "
                f"{', '.join(sorted(missing))}"
            )
        try:
            index = int(value["index"])
        except (TypeError, ValueError) as exc:
            raise OCRError(f"Predictions row {line_number} has a non-integer index") from exc
        if index in completed:
            raise OCRError(f"Duplicate OCR prediction index {index}")
        row = dict(value)
        row["index"] = index
        if value["model"] != MODEL_KEY:
            raise OCRError(
                f"Prediction {index} has model={value['model']!r}; expected {MODEL_KEY!r}"
            )
        row["_output_path"] = _resolve(str(value["output_path"]), path.parent)
        completed[index] = row
    return completed


def validate_resume(
    jobs: Sequence[dict[str, Any]],
    completed: dict[int, dict[str, Any]],
    *,
    checkpoint_sha256: str,
    config_sha256: str,
) -> None:
    by_index = {int(job["index"]): job for job in jobs}
    unexpected = sorted(set(completed).difference(by_index))
    if unexpected:
        raise OCRError(f"Predictions contain indices absent from the manifest: {unexpected}")
    for index, row in completed.items():
        if row["_output_path"] != by_index[index]["_output_path"]:
            raise OCRError(
                f"Prediction {index} refers to {row['_output_path']}, but its job "
                f"refers to {by_index[index]['_output_path']}"
            )
    try:
        validate_completed_provenance(completed, checkpoint_sha256, config_sha256)
    except OCRProvenanceError as exc:
        raise OCRError(str(exc)) from exc


def validate_generated_images(jobs: Sequence[dict[str, Any]]) -> None:
    from PIL import Image

    missing = [job["_output_path"] for job in jobs if not job["_output_path"].is_file()]
    if missing:
        preview = "\n  - ".join(str(path) for path in missing[:20])
        suffix = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise OCRError(
            f"OCR cannot start because {len(missing)} generated images are missing:\n"
            f"  - {preview}{suffix}"
        )
    for job in jobs:
        try:
            with Image.open(job["_output_path"]) as image:
                image.verify()
        except Exception as exc:
            raise OCRError(
                f"Generated image for job {job['index']} is unreadable: "
                f"{job['_output_path']}"
            ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recognize all v2 outputs with ABINet.")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    invocation_dir = Path.cwd()
    repository = _resolve(args.repository, invocation_dir)
    checkpoint = _resolve(args.checkpoint, invocation_dir)
    manifest = _resolve(args.manifest, invocation_dir)
    predictions = _resolve(args.predictions, invocation_dir)

    config_path = _require_file(
        repository / "configs" / "inference.yaml",
        "TextCtrl inference config",
    )
    _require_file(
        repository / "src" / "module" / "abinet" / "__init__.py",
        "TextCtrl ABINet package",
    )
    checkpoint = _require_file(checkpoint, "ABINet checkpoint")
    checkpoint_sha256 = sha256_file(checkpoint)
    config_sha256 = sha256_file(config_path)
    jobs = read_manifest(manifest)
    completed = {} if args.overwrite else read_predictions(predictions)
    validate_generated_images(jobs)
    validate_resume(
        jobs,
        completed,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
    )
    pending = [job for job in jobs if int(job["index"]) not in completed]
    if not pending:
        print(f"All {len(jobs)} OCR jobs are already complete.", flush=True)
        return 0

    try:
        from PIL import Image
        import torch
        import torchvision.transforms as transforms
        from omegaconf import OmegaConf
        from tqdm import tqdm
    except ImportError as exc:
        raise OCRError(f"OCR runtime dependency is unavailable: {exc}") from exc

    if not torch.cuda.is_available():
        raise OCRError(
            "ABINet OCR requires a CUDA GPU and a CUDA-enabled PyTorch installation"
        )

    predictions.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(repository)
    sys.path.insert(0, str(repository))
    try:
        from src.module.abinet import ABINetIterModel, CharsetMapper, postprocess
    except ImportError as exc:
        raise OCRError(f"Could not import TextCtrl ABINet from {repository}: {exc}") from exc

    config = OmegaConf.load("configs/inference.yaml").model.params.base_config.ocr_model
    model = ABINetIterModel(config).cuda()
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    charset = CharsetMapper(
        filename=str(config.charset_path), max_length=int(config.max_length) + 1
    )
    resize = transforms.Resize([int(config.height), int(config.width)])
    to_tensor = transforms.ToTensor()

    mode = "w" if args.overwrite or not predictions.is_file() else "a"
    with predictions.open(mode, encoding="utf-8", buffering=1) as output:
        for job in tqdm(pending, desc="ABINet OCR"):
            try:
                with Image.open(job["_output_path"]) as opened:
                    value = resize(to_tensor(opened.convert("RGB"))).unsqueeze(0).cuda()
                with torch.inference_mode():
                    prediction = postprocess(
                        model(value, mode="test"), charset, "alignment"
                    )[0][0]
            except Exception as exc:
                raise OCRError(
                    f"OCR failed for job {job['index']} image {job['_output_path']}"
                ) from exc
            row = {
                "index": int(job["index"]),
                "model": MODEL_KEY,
                "ocr_predicted_text": prediction,
                "output_path": str(job["_output_path"]),
                "image_sha256": sha256_file(job["_output_path"]),
                "ocr_checkpoint_sha256": checkpoint_sha256,
                "ocr_config_sha256": config_sha256,
            }
            output.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Completed {len(pending)} OCR job(s).", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except OCRError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
