"""Render the requested capitalization and punctuation comparison collages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .data import MODEL_KEYS
from .report import MODEL_DISPLAY


def load_font(path: str | Path | None, size: int) -> ImageFont.ImageFont:
    candidates = [path, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for candidate in candidates:
        if candidate in (None, ""):
            continue
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    return ImageFont.load_default()


def tile(
    image_path: str | Path,
    lines: Sequence[str],
    config: Mapping[str, Any],
) -> Image.Image:
    cell_width = int(config["cell_width"])
    image_height = int(config["image_height"])
    caption_height = int(config["caption_height"])
    font = load_font(config.get("font_path"), int(config.get("font_size", 13)))
    value = Image.new("RGB", (cell_width, caption_height + image_height), "white")
    draw = ImageDraw.Draw(value)
    line_height = max(14, int(config.get("font_size", 13)) + 3)
    for line_index, line in enumerate(lines):
        draw.text((7, 5 + line_index * line_height), str(line), fill="black", font=font)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    image.thumbnail((cell_width - 10, image_height - 10), Image.Resampling.LANCZOS)
    value.paste(
        image,
        (
            (cell_width - image.width) // 2,
            caption_height + (image_height - image.height) // 2,
        ),
    )
    return value


def metric_line(row: Mapping[str, Any]) -> str:
    return f"ACC: {row['ACC']:.0f}, NED: {row['NED']:.3f}, CER: {row['CER']:.3f}"


def result_lines(row: Mapping[str, Any]) -> list[str]:
    return [
        MODEL_DISPLAY[str(row["model"])],
        f"target: {row['target_text']}",
        metric_line(row),
    ]


def save_grid(grid: Sequence[Sequence[Image.Image]], destination: str | Path) -> Path:
    if not grid or not grid[0]:
        raise ValueError("A collage grid must not be empty")
    column_count = len(grid[0])
    if any(len(row) != column_count for row in grid):
        raise ValueError("Every collage row must have the same number of columns")
    tile_width, tile_height = grid[0][0].size
    canvas = Image.new(
        "RGB",
        (column_count * tile_width, len(grid) * tile_height),
        (224, 224, 224),
    )
    for row_index, row in enumerate(grid):
        for column_index, value in enumerate(row):
            canvas.paste(value, (column_index * tile_width, row_index * tile_height))
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def index_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in results:
        key = (str(row["model"]), int(row["sample_index"]), str(row["target_key"]))
        if key in indexed:
            raise ValueError(f"Duplicate result row for {key}")
        indexed[key] = row
    return indexed


def render_case_collage(
    samples: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    destination: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """Render five samples x (source, TextCtrl pair, SD3.5 pair)."""

    if len(samples) < 5:
        raise ValueError("The capitalization collage requires at least five samples")
    indexed = index_results(results)
    grid: list[list[Image.Image]] = []
    for sample in samples[:5]:
        sample_index = int(sample["sample_index"])
        row = [tile(sample["input_path"], [f"source: {sample['source_text']}"], config)]
        for model in MODEL_KEYS:
            for target_key in ("case_upper", "case_lower"):
                key = (model, sample_index, target_key)
                if key not in indexed:
                    raise ValueError(f"Missing collage result for {key}")
                result = indexed[key]
                row.append(tile(result["output_path"], result_lines(result), config))
        grid.append(row)
    return save_grid(grid, destination)


def render_special_collage(
    samples: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    destination: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """Render the first sample as two model rows x seven image columns."""

    if not samples:
        raise ValueError("The special-character collage requires a sample")
    sample = samples[0]
    sample_index = int(sample["sample_index"])
    indexed = index_results(results)
    grid: list[list[Image.Image]] = []
    for model in MODEL_KEYS:
        row = [
            tile(
                sample["input_path"],
                [MODEL_DISPLAY[model], f"source: {sample['source_text']}"],
                config,
            )
        ]
        for target_index in range(1, 7):
            key = (model, sample_index, f"special_{target_index}")
            if key not in indexed:
                raise ValueError(f"Missing collage result for {key}")
            result = indexed[key]
            row.append(tile(result["output_path"], result_lines(result), config))
        grid.append(row)
    return save_grid(grid, destination)

