"""Generate a 3x10 COCO-Text comparison collage for TextCtrl and SD3.5."""

from __future__ import annotations

import json
import os
import random
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
from PIL import Image, ImageDraw, ImageFont

from evaluate_coco_text import (
    _backend_config,
    _release_backend,
    _run_ocr,
    _target_candidates,
    annotation_mask,
)
from .inference_backend import SD35LoRABackend
from networks.scene_text_editing.configuration import resolve_path
from networks.scene_text_editing.datasets import EditingSample
from networks.scene_text_editing.evaluate_coco_text import (
    EvaluationError,
    _add_gaussian_noise,
    _load_metadata,
    _read_latest_predictions,
    _write_jsonl,
    choose_target_text,
    expanded_crop_box,
    text_metrics,
)


def _prepare_shared_samples(
    images: Sequence[dict[str, Any]],
    all_images: Sequence[dict[str, Any]],
    dataset_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[EditingSample]]:
    source_dir = output_dir / "source_crops"
    noisy_dir = output_dir / "noisy_source_crops"
    mask_dir = output_dir / "masks"
    textctrl_dir = output_dir / "textctrl_generated"
    sd35_dir = output_dir / "sd35_generated"
    for directory in (source_dir, noisy_dir, mask_dir, textctrl_dir, sd35_dir):
        directory.mkdir(parents=True, exist_ok=True)

    target_rng = random.Random(int(config["target_seed"]))
    noise_rng = random.Random(int(config["noise"]["seed"]))
    candidates = _target_candidates(all_images)
    crop_config = config["crop"]
    noise_config = config["noise"]
    records: list[dict[str, Any]] = []
    samples: list[EditingSample] = []
    for index, image_record in enumerate(images):
        instances = image_record.get("instances", [])
        if not instances:
            raise EvaluationError(f"Image record {image_record!r} has no annotation.")
        instance = instances[0]
        source_text = str(instance["text"])
        target_text = choose_target_text(source_text, candidates, target_rng)
        noise_std = noise_rng.uniform(
            float(noise_config["minimum_standard_deviation"]),
            float(noise_config["maximum_standard_deviation"]),
        )
        image_title = str(image_record["file_name"])
        image_path = dataset_dir / "image" / image_title
        if not image_path.is_file():
            raise EvaluationError(f"Missing COCO-Text image: {image_path}")
        stem = f"{index:02d}_{Path(image_title).stem}"
        source_path = source_dir / f"{stem}.png"
        noisy_path = noisy_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}.png"
        textctrl_path = textctrl_dir / f"{stem}.png"
        sd35_path = sd35_dir / f"{stem}.png"

        with Image.open(image_path) as opened:
            source_image = opened.convert("RGB")
            crop_box = expanded_crop_box(
                instance["bbox"],
                source_image.size,
                float(crop_config["padding_ratio"]),
                int(crop_config["minimum_padding_pixels"]),
            )
            noisy_image = _add_gaussian_noise(
                source_image,
                noise_std,
                int(noise_config["seed"]) + index,
            )
            mask = annotation_mask(source_image.size, instance)
            source_image.crop(crop_box).save(source_path)
            noisy_image.crop(crop_box).save(noisy_path)
            mask.crop(crop_box).save(mask_path)

        records.append(
            {
                "index": index,
                "image_title": image_title,
                "source_text": source_text,
                "target_text": target_text,
                "noise_standard_deviation": noise_std,
                "input_path": str(noisy_path),
                "mask_path": str(mask_path),
                "textctrl_output_path": str(textctrl_path),
                "sd35_output_path": str(sd35_path),
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
    return records, samples


def _model_manifest(
    records: Sequence[Mapping[str, Any]], model: str
) -> list[dict[str, Any]]:
    output_key = f"{model}_output_path"
    return [
        {
            "index": int(record["index"]),
            "image_title": record["image_title"],
            "source_text": record["source_text"],
            "target_text": record["target_text"],
            "input_path": record["input_path"],
            "mask_path": record["mask_path"],
            "output_path": record[output_key],
        }
        for record in records
    ]


def _run_textctrl(
    config: Mapping[str, Any], manifest_path: Path, predictions_path: Path
) -> None:
    textctrl = config["textctrl"]
    repository = resolve_path(str(textctrl["repository_dir"]))
    worker = PROJECT_ROOT / "networks/scene_text_editing/textctrl_coco_eval_worker.py"
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


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((center_x - (box[2] - box[0]) // 2, y), text, font=font, fill=fill)


def _paste_contained(
    canvas: Image.Image,
    image_path: Path,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    scale = min((right - left) / image.width, (bottom - top) / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(size, Image.Resampling.LANCZOS)
    offset = (
        left + (right - left - size[0]) // 2,
        top + (bottom - top - size[1]) // 2,
    )
    canvas.paste(image, offset)


def _render_collage(
    records: Sequence[Mapping[str, Any]],
    textctrl_predictions: Mapping[int, Mapping[str, Any]],
    sd35_predictions: Mapping[int, Mapping[str, Any]],
    destination: Path,
    config: Mapping[str, Any],
) -> None:
    collage_config = config["collage"]
    cell_width = int(collage_config["cell_width"])
    image_height = int(collage_config["image_height"])
    caption_height = int(collage_config["caption_height"])
    padding = int(collage_config["cell_padding"])
    border = int(collage_config["border_width"])
    row_height = caption_height + image_height + 2 * padding
    canvas = Image.new(
        "RGB", (cell_width * len(records), row_height * 3), color=(245, 246, 248)
    )
    draw = ImageDraw.Draw(canvas)
    regular = _load_font(str(collage_config["font_path"]), 15)
    small = _load_font(str(collage_config["font_path"]), 12)
    bold = _load_font(str(collage_config["bold_font_path"]), 17)
    colors = ((239, 244, 250), (246, 249, 244), (250, 246, 240))

    for row_index in range(3):
        for column, record in enumerate(records):
            left = column * cell_width
            top = row_index * row_height
            right = left + cell_width
            bottom = top + row_height
            draw.rectangle(
                (left, top, right - 1, bottom - 1),
                fill=colors[row_index],
                outline=(170, 176, 184),
                width=border,
            )
            center_x = left + cell_width // 2
            if row_index == 0:
                caption = (
                    f"Noise sigma: {float(record['noise_standard_deviation']):.1f}/255"
                )
                _centered_text(draw, center_x, top + 28, caption, bold, (32, 43, 56))
                image_path = Path(str(record["input_path"]))
            else:
                predictions = textctrl_predictions if row_index == 1 else sd35_predictions
                prediction = str(predictions[int(record["index"])]["ocr_predicted_text"])
                metrics = text_metrics(str(record["target_text"]), prediction)
                _centered_text(
                    draw,
                    center_x,
                    top + 6,
                    f"GT: {record['target_text']}",
                    bold,
                    (23, 35, 48),
                )
                shown_prediction = prediction if prediction else "<empty>"
                _centered_text(
                    draw,
                    center_x,
                    top + 29,
                    f"OCR: {shown_prediction}",
                    regular,
                    (45, 54, 65),
                )
                metric_text = (
                    f"ACC {metrics['ACC']}  NED {metrics['NED']:.2f}  "
                    f"CER {metrics['CER']:.2f}"
                )
                _centered_text(
                    draw, center_x, top + 53, metric_text, small, (63, 70, 80)
                )
                key = "textctrl_output_path" if row_index == 1 else "sd35_output_path"
                image_path = Path(str(record[key]))
            image_box = (
                left + padding,
                top + caption_height + padding,
                right - padding,
                bottom - padding,
            )
            draw.rectangle(image_box, fill=(0, 0, 0))
            _paste_contained(canvas, image_path, image_box)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


@hydra.main(version_base="1.3", config_path=".", config_name="create_coco_text_collage")
def main(config: DictConfig) -> None:
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise EvaluationError("Hydra did not compose a mapping config.")
    sample_count = int(resolved["sample_count"])
    if sample_count != 10:
        raise EvaluationError("This comparison collage requires sample_count=10.")
    dataset_dir = resolve_path(str(resolved["dataset_dir"]))
    output_dir = resolve_path(str(resolved["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_images = _load_metadata(dataset_dir / str(resolved["labels_file"]))
    images = all_images[:sample_count]
    records, samples = _prepare_shared_samples(
        images, all_images, dataset_dir, output_dir, resolved
    )

    shared_manifest = output_dir / "collage_manifest.jsonl"
    textctrl_manifest_path = output_dir / "textctrl_manifest.jsonl"
    sd35_manifest_path = output_dir / "sd35_manifest.jsonl"
    textctrl_predictions_path = output_dir / "textctrl_ocr_predictions.jsonl"
    sd35_predictions_path = output_dir / "sd35_ocr_predictions.jsonl"
    textctrl_manifest = _model_manifest(records, "textctrl")
    sd35_manifest = _model_manifest(records, "sd35")
    _write_jsonl(shared_manifest, records)
    _write_jsonl(textctrl_manifest_path, textctrl_manifest)
    _write_jsonl(sd35_manifest_path, sd35_manifest)
    (output_dir / "config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True), encoding="utf-8"
    )

    _run_textctrl(resolved, textctrl_manifest_path, textctrl_predictions_path)
    needs_sd35 = bool(resolved.get("overwrite", False)) or any(
        not Path(record["sd35_output_path"]).is_file() for record in records
    )
    if needs_sd35:
        backend = SD35LoRABackend(_backend_config(dict(resolved)))
        try:
            backend.run(
                samples,
                output_dir / "sd35_generated",
                overwrite=bool(resolved.get("overwrite", False)),
            )
        finally:
            _release_backend(backend)
    _run_ocr(resolved, sd35_manifest_path, sd35_predictions_path)

    textctrl_predictions = _read_latest_predictions(textctrl_predictions_path)
    sd35_predictions = _read_latest_predictions(sd35_predictions_path)
    if len(textctrl_predictions) != sample_count or len(sd35_predictions) != sample_count:
        raise EvaluationError("OCR predictions are incomplete; collage was not rendered.")
    destination = output_dir / str(resolved["collage_filename"])
    _render_collage(
        records,
        textctrl_predictions,
        sd35_predictions,
        destination,
        resolved,
    )
    summary = {
        "sample_count": sample_count,
        "textctrl_exact_matches": sum(
            text_metrics(
                str(record["target_text"]),
                str(textctrl_predictions[int(record["index"])]["ocr_predicted_text"]),
            )["ACC"]
            for record in records
        ),
        "sd35_exact_matches": sum(
            text_metrics(
                str(record["target_text"]),
                str(sd35_predictions[int(record["index"])]["ocr_predicted_text"]),
            )["ACC"]
            for record in records
        ),
        "collage_path": str(destination),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
