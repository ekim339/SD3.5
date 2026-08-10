"""Standalone Hydra inference entry point for the fine-tuned SD3.5 editor."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .inference_backend import SD35LoRABackend
from networks.scene_text_editing.datasets import EditingSample, load_dataset


def _backend_config(config: DictConfig) -> dict:
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    return {
        **resolved,
        "task": {
            "name": "text_image_editing",
            "dataset": resolved["dataset"],
            "prompts": {"name": "sd35_scene_text", **resolved["prompt"]},
            "generation": {
                **resolved["generation"],
                "width": resolved["network"]["width"],
                "height": resolved["network"]["height"],
            },
        },
    }


def _load_samples(resolved: dict) -> list[EditingSample]:
    input_config = resolved["input"]
    mode = str(input_config.get("mode", "dataset"))
    if mode == "dataset":
        return list(load_dataset(resolved["dataset"]))
    if mode != "direct":
        raise ValueError("input.mode must be either direct or dataset")
    required = ("source_image", "source_text", "target_text")
    missing = [name for name in required if not input_config.get(name)]
    if missing:
        raise ValueError(f"Direct input is missing: {', '.join(missing)}")
    source = Path(str(input_config["source_image"])).expanduser().resolve()
    mask_value = input_config.get("mask_image")
    mask = Path(str(mask_value)).expanduser().resolve() if mask_value else None
    if not source.is_file():
        raise FileNotFoundError(f"Input image does not exist: {source}")
    if mask is not None and not mask.is_file():
        raise FileNotFoundError(f"Input mask does not exist: {mask}")
    return [
        EditingSample(
            sample_id=str(input_config.get("sample_id") or source.stem),
            source_image=source,
            source_text=str(input_config["source_text"]),
            target_text=str(input_config["target_text"]),
            mask_image=mask,
        )
    ]


@hydra.main(version_base="1.3", config_path=".", config_name="inference")
def main(config: DictConfig) -> None:
    resolved = _backend_config(config)
    checkpoint = Path(resolved["network"]["checkpoint_path"])
    if bool(resolved.get("validate_only", False)):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        _load_samples(resolved)
        print(OmegaConf.to_yaml(config, resolve=True))
        print("Configuration is valid; no models were loaded.")
        return

    samples = _load_samples(resolved)
    limit = resolved.get("limit")
    if limit is not None:
        samples = samples[: int(limit)]
    if not samples:
        raise RuntimeError("The selected input contains no samples.")
    output_dir = Path(resolved["output_dir"]).resolve()
    records = SD35LoRABackend(resolved).run(
        samples, output_dir, overwrite=bool(resolved.get("overwrite", False))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True), encoding="utf-8"
    )
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    generated = sum(record["status"] == "generated" for record in records)
    print(f"Completed {generated} edit(s). Results: {output_dir}")


if __name__ == "__main__":
    main()
