"""Hydra entrypoint for text-image-editing inference."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from scene_text_editing.configuration import resolve_path, validate_config
from scene_text_editing.datasets import DatasetError, load_dataset
from scene_text_editing.networks import create_backend


class InferenceError(RuntimeError):
    """Raised when a composed inference job cannot be completed."""


def _write_run_artifacts(
    output_dir: Path,
    config: DictConfig,
    records: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True),
        encoding="utf-8",
    )
    manifest = output_dir / "results.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len({record.get("sample_id") for record in records}),
        "output_count": sum(
            record.get("status") == "generated" for record in records
        ),
        "skipped_count": sum(
            record.get("status") == "skipped_existing" for record in records
        ),
        "results_manifest": str(manifest),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="inference")
def main(config: DictConfig) -> None:
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise InferenceError("Hydra did not compose a mapping config.")
    validate_config(resolved)
    if bool(resolved.get("validate_only", False)):
        print(OmegaConf.to_yaml(config, resolve=True))
        print("Configuration is valid; no models or dataset were loaded.")
        return

    try:
        dataset = load_dataset(resolved["task"]["dataset"])
    except DatasetError as exc:
        raise InferenceError(f"Could not load the selected dataset: {exc}") from exc
    limit_value = resolved.get("limit")
    if limit_value is not None:
        limit = int(limit_value)
        if limit <= 0:
            raise InferenceError("limit must be null or a positive integer.")
        samples = list(dataset[:limit])
    else:
        samples = list(dataset)
    if not samples:
        raise InferenceError("The selected dataset contains no editing samples.")

    output_dir = resolve_path(str(resolved["output_dir"]))
    backend = create_backend(resolved)
    records = backend.run(
        samples,
        output_dir,
        overwrite=bool(resolved.get("overwrite", False)),
    )
    _write_run_artifacts(output_dir, config, records)
    generated = sum(record.get("status") == "generated" for record in records)
    print(f"Completed {generated} edit(s). Results: {output_dir}")


if __name__ == "__main__":
    main()
