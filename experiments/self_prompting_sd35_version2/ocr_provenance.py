"""Per-prediction content fingerprints for safe resumable OCR."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from .provenance import sha256_file
except ImportError:  # Direct worker execution.
    from provenance import sha256_file  # type: ignore[no-redef]


class OCRProvenanceError(RuntimeError):
    """An existing OCR row does not match its current inputs."""


def metadata(image: Path, checkpoint: Path, config: Path) -> dict[str, str]:
    """Return the material digests stored with one OCR prediction."""

    try:
        return {
            "image_sha256": sha256_file(image),
            "ocr_checkpoint_sha256": sha256_file(checkpoint),
            "ocr_config_sha256": sha256_file(config),
        }
    except OSError as exc:
        raise OCRProvenanceError(f"Could not fingerprint OCR input: {exc}") from exc


def validate_completed(
    completed: Mapping[int, Mapping[str, Any]],
    checkpoint_sha256: str,
    config_sha256: str,
) -> None:
    """Reject OCR rows created from other pixels, weights, or configuration."""

    for index, row in completed.items():
        if str(row["ocr_checkpoint_sha256"]) != checkpoint_sha256:
            raise OCRProvenanceError(
                f"OCR checkpoint changed for prediction {index}; rerun with --overwrite"
            )
        if str(row["ocr_config_sha256"]) != config_sha256:
            raise OCRProvenanceError(
                f"OCR configuration changed for prediction {index}; rerun with --overwrite"
            )
        try:
            current_image_sha256 = sha256_file(Path(row["_output_path"]))
        except OSError as exc:
            raise OCRProvenanceError(
                f"Could not fingerprint generated image for prediction {index}: {exc}"
            ) from exc
        if str(row["image_sha256"]) != current_image_sha256:
            raise OCRProvenanceError(
                f"Generated image changed for prediction {index}; rerun with --overwrite"
            )
