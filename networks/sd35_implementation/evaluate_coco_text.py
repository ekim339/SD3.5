"""Evaluate the fine-tuned SD3.5 editor on the local 1K COCO-Text subset."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

from .inference_backend import SD35LoRABackend
from networks.scene_text_editing.configuration import resolve_path
from networks.scene_text_editing.datasets import EditingSample
from networks.scene_text_editing.evaluate_coco_text import (
    EvaluationError,
    _add_gaussian_noise,
    _load_metadata,
    _read_latest_predictions,
    _write_csv,
    _write_jsonl,
    choose_target_text,
    expanded_crop_box,
)


def annotation_mask(
    image_size: tuple[int, int],
    instance: Mapping[str, Any],
) -> Image.Image:
    """Rasterize only the selected COCO-Text annotation as a binary mask."""

    mask = Image.new("L", image_size, color=0)
    draw = ImageDraw.Draw(mask)
    polygon = instance.get("polygon")
    if isinstance(polygon, Sequence) and len(polygon) >= 3:
        points = [(float(point[0]), float(point[1])) for point in polygon]
        draw.polygon(points, fill=255)
    else:
        bbox = instance.get("bbox")
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise EvaluationError("Selected annotation has no polygon or bbox.")
        x, y, width, height = (float(value) for value in bbox)
        draw.rectangle((x, y, x + width, y + height), fill=255)
    return mask


def _target_candidates(images: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        {
            str(instance["text"])
            for image in images
            for instance in image.get("instances", [])
            if len(str(instance.get("text", ""))) == 5
        },
        key=lambda value: (value.casefold(), value),
    )


def _prepare_samples(
    images: Sequence[dict[str, Any]],
    all_images: Sequence[dict[str, Any]],
    dataset_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[EditingSample]]:
    import random

    source_dir = output_dir / "source_crops"
    noisy_dir = output_dir / "noisy_source_crops"
    mask_dir = output_dir / "masks"
    generated_dir = output_dir / "generated_crops"
    for directory in (source_dir, noisy_dir, mask_dir, generated_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    target_rng = random.Random(seed)
    candidates = _target_candidates(all_images)
    crop_config = config["crop"]
    noise_std = float(config["noise"]["standard_deviation"])
    manifest: list[dict[str, Any]] = []
    samples: list[EditingSample] = []
    for index, record in enumerate(images):
        instances = record.get("instances", [])
        if not instances:
            raise EvaluationError(f"Image record {record!r} has no text instance.")
        instance = instances[0]
        source_text = str(instance["text"])
        target_text = choose_target_text(source_text, candidates, target_rng)
        image_title = str(record["file_name"])
        image_path = dataset_dir / "image" / image_title
        if not image_path.is_file():
            raise EvaluationError(f"Missing COCO-Text image: {image_path}")
        stem = f"{index:04d}_{Path(image_title).stem}"
        source_path = source_dir / f"{stem}.png"
        noisy_path = noisy_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}.png"
        output_path = generated_dir / f"{stem}.png"

        with Image.open(image_path) as opened:
            source_image = opened.convert("RGB")
            crop_box = expanded_crop_box(
                instance["bbox"],
                source_image.size,
                float(crop_config["padding_ratio"]),
                int(crop_config["minimum_padding_pixels"]),
            )
            noisy_image = _add_gaussian_noise(source_image, noise_std, seed + index)
            selected_mask = annotation_mask(source_image.size, instance)
            source_image.crop(crop_box).save(source_path)
            noisy_image.crop(crop_box).save(noisy_path)
            selected_mask.crop(crop_box).save(mask_path)

        manifest.append(
            {
                "index": index,
                "image_title": image_title,
                "source_text": source_text,
                "target_text": target_text,
                "input_path": str(noisy_path),
                "mask_path": str(mask_path),
                "output_path": str(output_path),
                "crop_box": list(crop_box),
            }
        )
        samples.append(
            EditingSample(
                sample_id=stem,
                source_image=noisy_path,
                source_text=source_text,
                target_text=target_text,
                mask_image=mask_path,
            )
        )
    return manifest, samples


def _backend_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **config,
        "task": {
            "name": "text_image_editing",
            "dataset": {},
            "prompts": {"name": "sd35_coco_text", **config["prompt"]},
            "generation": {
                **config["generation"],
                "width": config["network"]["width"],
                "height": config["network"]["height"],
            },
        },
    }


def _release_backend(backend: SD35LoRABackend) -> None:
    torch = backend.torch
    backend.pipe = None
    backend.model = None
    del backend
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_ocr(
    config: Mapping[str, Any],
    manifest_path: Path,
    predictions_path: Path,
) -> None:
    ocr = config["ocr"]
    repository = resolve_path(str(ocr["repository_dir"]))
    worker = Path(__file__).with_name("abinet_ocr_worker.py").resolve()
    command = [
        str(ocr["python_executable"]),
        str(worker),
        "--repository",
        str(repository),
        "--checkpoint",
        str(resolve_path(str(ocr["checkpoint_path"]))),
        "--manifest",
        str(manifest_path),
        "--predictions",
        str(predictions_path),
    ]
    if bool(config.get("overwrite", False)):
        command.append("--overwrite")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(repository), environment.get("PYTHONPATH", ""))
        if value
    )
    subprocess.run(command, cwd=repository, env=environment, check=True)


@hydra.main(version_base="1.3", config_path=".", config_name="evaluate_coco_text")
def main(config: DictConfig) -> None:
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise EvaluationError("Hydra did not compose a mapping config.")
    dataset_dir = resolve_path(str(resolved["dataset_dir"]))
    output_dir = resolve_path(str(resolved["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_images = _load_metadata(dataset_dir / str(resolved["labels_file"]))
    images = all_images
    limit = resolved.get("limit")
    if limit is not None:
        limit = int(limit)
        if limit <= 0:
            raise EvaluationError("limit must be null or a positive integer.")
        images = images[:limit]

    manifest, samples = _prepare_samples(
        images, all_images, dataset_dir, output_dir, resolved
    )
    manifest_path = output_dir / "evaluation_manifest.jsonl"
    predictions_path = output_dir / "ocr_predictions.jsonl"
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"
    _write_jsonl(manifest_path, manifest)
    (output_dir / "config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True), encoding="utf-8"
    )

    generated_dir = output_dir / "generated_crops"
    needs_generation = bool(resolved.get("overwrite", False)) or any(
        not Path(record["output_path"]).is_file() for record in manifest
    )
    try:
        if needs_generation:
            backend = SD35LoRABackend(_backend_config(dict(resolved)))
            try:
                backend.run(
                    samples,
                    generated_dir,
                    overwrite=bool(resolved.get("overwrite", False)),
                )
            finally:
                _release_backend(backend)
        _run_ocr(resolved, manifest_path, predictions_path)
    finally:
        predictions = _read_latest_predictions(predictions_path)
        summary = _write_csv(csv_path, manifest, predictions)
        summary.update(
            {
                "requested_images": len(manifest),
                "complete": len(predictions) == len(manifest),
                "checkpoint": str(resolve_path(str(resolved["network"]["checkpoint_path"]))),
                "results_csv": str(csv_path),
            }
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

