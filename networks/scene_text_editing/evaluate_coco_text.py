"""Evaluate released TextCtrl SD1.5 on the local 1K COCO-Text subset."""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from networks.scene_text_editing.configuration import resolve_path


CSV_FIELDS = (
    "image_title",
    "ground_truth_text",
    "ocr_predicted_text",
    "ACC",
    "NED",
    "CER",
)


class EvaluationError(RuntimeError):
    """Raised when the COCO-Text evaluation cannot be completed."""


def levenshtein_distance(reference: str, prediction: str) -> int:
    """Return character-level Levenshtein distance using linear memory."""

    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for row, reference_character in enumerate(reference, start=1):
        current = [row]
        for column, prediction_character in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + (reference_character != prediction_character),
                )
            )
        previous = current
    return previous[-1]


def text_metrics(ground_truth: str, prediction: str) -> dict[str, float | int]:
    """Compute case-insensitive exact accuracy, NED similarity, and CER."""

    reference = ground_truth.casefold()
    hypothesis = prediction.casefold()
    distance = levenshtein_distance(reference, hypothesis)
    return {
        "ACC": int(reference == hypothesis),
        "NED": 1.0 - distance / max(len(reference), len(hypothesis), 1),
        "CER": distance / max(len(reference), 1),
    }


def expanded_crop_box(
    bbox: Sequence[float],
    image_size: tuple[int, int],
    padding_ratio: float,
    minimum_padding: int,
) -> tuple[int, int, int, int]:
    """Expand a COCO ``[x, y, width, height]`` box and clip to the image."""

    if len(bbox) != 4:
        raise EvaluationError(f"Expected a four-value bbox, received {bbox!r}.")
    x, y, width, height = (float(value) for value in bbox)
    if width <= 0 or height <= 0:
        raise EvaluationError(f"Invalid non-positive bbox {bbox!r}.")
    pad_x = max(int(round(width * padding_ratio)), minimum_padding)
    pad_y = max(int(round(height * padding_ratio)), minimum_padding)
    image_width, image_height = image_size
    left = max(0, int(np.floor(x)) - pad_x)
    top = max(0, int(np.floor(y)) - pad_y)
    right = min(image_width, int(np.ceil(x + width)) + pad_x)
    bottom = min(image_height, int(np.ceil(y + height)) + pad_y)
    if right <= left or bottom <= top:
        raise EvaluationError(f"BBox {bbox!r} produced an empty crop.")
    return left, top, right, bottom


def choose_target_text(
    source_text: str,
    candidates: Sequence[str],
    rng: random.Random,
) -> str:
    """Choose a five-character target that differs case-insensitively."""

    valid = [
        candidate
        for candidate in candidates
        if len(candidate) == 5 and candidate.casefold() != source_text.casefold()
    ]
    if not valid:
        raise EvaluationError("No target text differs from the source text.")
    return rng.choice(valid)


def _load_metadata(labels_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not read {labels_path}: {exc}") from exc
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise EvaluationError(f"{labels_path} does not contain a non-empty images list.")
    return images


def _add_gaussian_noise(image: Image.Image, standard_deviation: float, seed: int) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    if standard_deviation > 0:
        generator = np.random.default_rng(seed)
        array += generator.normal(0.0, standard_deviation, size=array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def _prepare_samples(
    images: Sequence[dict[str, Any]],
    dataset_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    target_candidates: Sequence[str],
) -> list[dict[str, Any]]:
    original_dir = output_dir / "source_crops"
    noisy_dir = output_dir / "noisy_source_crops"
    generated_dir = output_dir / "generated_crops"
    for directory in (original_dir, noisy_dir, generated_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    target_rng = random.Random(seed)
    crop_config = config["crop"]
    noise_standard_deviation = float(config["noise"]["standard_deviation"])
    samples: list[dict[str, Any]] = []
    for index, record in enumerate(images):
        instances = record.get("instances", [])
        if not instances:
            raise EvaluationError(f"Image record {record!r} has no text instance.")
        instance = instances[0]
        source_text = str(instance["text"])
        target_text = choose_target_text(source_text, target_candidates, target_rng)
        image_title = str(record["file_name"])
        source_path = dataset_dir / "image" / image_title
        if not source_path.is_file():
            raise EvaluationError(f"Missing COCO-Text image: {source_path}")
        stem = f"{index:04d}_{Path(image_title).stem}"
        original_path = original_dir / f"{stem}.png"
        noisy_path = noisy_dir / f"{stem}.png"
        generated_path = generated_dir / f"{stem}.png"

        with Image.open(source_path) as opened:
            source_image = opened.convert("RGB")
            noisy_image = _add_gaussian_noise(
                source_image,
                noise_standard_deviation,
                seed + index,
            )
            crop_box = expanded_crop_box(
                instance["bbox"],
                source_image.size,
                float(crop_config["padding_ratio"]),
                int(crop_config["minimum_padding_pixels"]),
            )
            source_image.crop(crop_box).save(original_path)
            noisy_image.crop(crop_box).save(noisy_path)

        samples.append(
            {
                "index": index,
                "image_title": image_title,
                "source_text": source_text,
                "target_text": target_text,
                "input_path": str(noisy_path),
                "output_path": str(generated_path),
                "crop_box": list(crop_box),
            }
        )
    return samples


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_latest_predictions(path: Path) -> dict[int, dict[str, Any]]:
    predictions: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return predictions
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            predictions[int(record["index"])] = record
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EvaluationError(
                f"Invalid prediction at {path}:{line_number}: {exc}"
            ) from exc
    return predictions


def _write_csv(
    path: Path,
    samples: Sequence[Mapping[str, Any]],
    predictions: Mapping[int, Mapping[str, Any]],
) -> dict[str, float | int]:
    rows = []
    for sample in samples:
        index = int(sample["index"])
        if index not in predictions:
            continue
        prediction = str(predictions[index]["ocr_predicted_text"])
        ground_truth = str(sample["target_text"])
        metrics = text_metrics(ground_truth, prediction)
        rows.append(
            {
                "image_title": sample["image_title"],
                "ground_truth_text": ground_truth,
                "ocr_predicted_text": prediction,
                "ACC": metrics["ACC"],
                "NED": f"{metrics['NED']:.6f}",
                "CER": f"{metrics['CER']:.6f}",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    count = len(rows)
    return {
        "evaluated_images": count,
        "ACC": sum(float(row["ACC"]) for row in rows) / max(count, 1),
        "NED": sum(float(row["NED"]) for row in rows) / max(count, 1),
        "CER": sum(float(row["CER"]) for row in rows) / max(count, 1),
    }


def _run_worker(
    config: Mapping[str, Any],
    manifest_path: Path,
    predictions_path: Path,
) -> None:
    textctrl = config["textctrl"]
    repository = resolve_path(str(textctrl["repository_dir"]))
    worker = Path(__file__).with_name("textctrl_coco_eval_worker.py").resolve()
    command = [
        str(textctrl["python_executable"]),
        str(worker),
        "--repository",
        str(repository),
        "--checkpoint",
        str(resolve_path(str(textctrl["checkpoint_path"]))),
        "--ocr-checkpoint",
        str(resolve_path(str(textctrl["ocr_checkpoint_path"]))),
        "--manifest",
        str(manifest_path),
        "--predictions",
        str(predictions_path),
        "--seed",
        str(int(config["seed"])),
        "--starting-layer",
        str(int(textctrl["starting_layer"])),
        "--num-inference-steps",
        str(int(textctrl["num_inference_steps"])),
        "--guidance-scale",
        str(float(textctrl["guidance_scale"])),
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


@hydra.main(version_base="1.3", config_path="configs", config_name="evaluate_coco_text")
def main(config: DictConfig) -> None:
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise EvaluationError("Hydra did not compose a mapping config.")
    dataset_dir = resolve_path(str(resolved["dataset_dir"]))
    output_dir = resolve_path(str(resolved["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_images = _load_metadata(dataset_dir / str(resolved["labels_file"]))
    target_candidates = sorted(
        {
            str(instance["text"])
            for image in all_images
            for instance in image.get("instances", [])
            if len(str(instance.get("text", ""))) == 5
        },
        key=lambda value: (value.casefold(), value),
    )
    images = all_images
    limit = resolved.get("limit")
    if limit is not None:
        limit = int(limit)
        if limit <= 0:
            raise EvaluationError("limit must be null or a positive integer.")
        images = images[:limit]
    samples = _prepare_samples(
        images,
        dataset_dir,
        output_dir,
        resolved,
        target_candidates,
    )
    manifest_path = output_dir / "evaluation_manifest.jsonl"
    predictions_path = output_dir / "ocr_predictions.jsonl"
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"
    _write_jsonl(manifest_path, samples)
    (output_dir / "config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True), encoding="utf-8"
    )

    try:
        _run_worker(resolved, manifest_path, predictions_path)
    finally:
        predictions = _read_latest_predictions(predictions_path)
        summary = _write_csv(csv_path, samples, predictions)
        summary.update(
            {
                "requested_images": len(samples),
                "complete": len(predictions) == len(samples),
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

