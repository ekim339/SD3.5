"""Generation signatures that make path-based resumption scientifically safe."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

try:
    from .provenance import read_json_object, sha256_file, write_json_object_atomic
except ImportError:  # Direct worker execution.
    from provenance import (  # type: ignore[no-redef]
        read_json_object,
        sha256_file,
        write_json_object_atomic,
    )


PROVENANCE_FILENAME = "generation_provenance.json"
SCHEMA_VERSION = 1


class GenerationProvenanceError(RuntimeError):
    """Existing outputs cannot be shown to match the requested generation."""


def _package_versions() -> dict[str, str]:
    values: dict[str, str] = {}
    for package in ("diffusers", "torch", "peft", "safetensors"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "not-installed"
    return values


def _digests(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in sorted(set(paths))}


def build_signature(args: Any, jobs: Sequence[Any], worker_file: Path) -> dict[str, Any]:
    """Fingerprint every material input to deterministic image generation."""

    worker_file = worker_file.resolve()
    project_root = worker_file.parents[2]
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    runtime_sources = [
        worker_file,
        Path(__file__).resolve(),
        Path(__file__).with_name("provenance.py").resolve(),
        project_root / "CODEX" / "self_prompting_sd35" / "conditioning.py",
        project_root / "CODEX" / "self_prompting_sd35" / "dataset.py",
        project_root / "CODEX" / "self_prompting_sd35" / "model.py",
    ]
    font_path = None if args.font_path is None else Path(args.font_path).resolve()
    font = None
    if font_path is not None:
        font = {"path": str(font_path), "sha256": sha256_file(font_path)}
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "checkpoint": {
            "path": str(checkpoint),
            "pytorch_lora_weights_sha256": sha256_file(
                checkpoint / "pytorch_lora_weights.safetensors"
            ),
            "input_projection_sha256": sha256_file(
                checkpoint / "input_projection.safetensors"
            ),
        },
        "inference": {
            "base_model": str(args.base_model),
            "revision": getattr(args, "revision", None),
            "resolution": int(args.resolution),
            "steps": int(args.steps),
            "max_sequence_length": int(args.max_sequence_length),
            "dtype": str(args.dtype),
            "font": font,
        },
        "input_images": _digests([Path(job.input_path).resolve() for job in jobs]),
        "mask_images": _digests([Path(job.mask_path).resolve() for job in jobs]),
        "runtime_sources": _digests(runtime_sources),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "packages": _package_versions(),
        },
    }


def begin(
    args: Any, jobs: Sequence[Any], worker_file: Path
) -> tuple[Path, dict[str, Any]]:
    """Validate any existing outputs, then record an in-progress signature."""

    manifest = Path(args.manifest).expanduser().resolve()
    state_path = manifest.parent / PROVENANCE_FILENAME
    try:
        signature = build_signature(args, jobs, worker_file)
    except (OSError, ValueError) as exc:
        raise GenerationProvenanceError(
            f"Could not fingerprint generation inputs: {exc}"
        ) from exc
    existing_outputs = any(Path(job.output_path).exists() for job in jobs)
    previous: dict[str, Any] | None = None
    if state_path.exists():
        try:
            previous = read_json_object(state_path)
        except ValueError as exc:
            if existing_outputs and not args.overwrite:
                raise GenerationProvenanceError(
                    f"Existing outputs have unreadable provenance {state_path}; "
                    "rerun with --overwrite"
                ) from exc
    if existing_outputs and not args.overwrite:
        if previous is None:
            raise GenerationProvenanceError(
                f"Existing outputs have no provenance file {state_path}; "
                "rerun with --overwrite"
            )
        if previous.get("signature") != signature:
            raise GenerationProvenanceError(
                "Existing outputs were generated with different inputs, code, "
                "checkpoint, or inference settings; rerun with --overwrite"
            )
        if previous.get("unsafe_resume") and not previous.get("complete"):
            raise GenerationProvenanceError(
                "A prior overwrite run was interrupted while old outputs were still "
                "present; rerun with --overwrite to avoid mixing configurations"
            )
    state = {
        "schema_version": SCHEMA_VERSION,
        "complete": False,
        "unsafe_resume": bool(existing_outputs and args.overwrite),
        "signature": signature,
    }
    try:
        write_json_object_atomic(state_path, state)
    except OSError as exc:
        raise GenerationProvenanceError(
            f"Could not write generation provenance {state_path}: {exc}"
        ) from exc
    return state_path, signature


def finish(state_path: Path, signature: dict[str, Any]) -> None:
    """Mark a fully completed generation signature as safe to reuse."""

    try:
        write_json_object_atomic(
            state_path,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "unsafe_resume": False,
                "signature": signature,
            },
        )
    except OSError as exc:
        raise GenerationProvenanceError(
            f"Could not finalize generation provenance {state_path}: {exc}"
        ) from exc
