"""Render the exact 3x5 case and 7x1 special-character collages."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .data import SPECIAL_TARGET_KEYS


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
    """Draw captions above an aspect-preserving image tile."""

    source = Path(image_path)
    if not source.is_file():
        raise FileNotFoundError(f"Missing collage image: {source}")
    cell_width = int(config["cell_width"])
    image_height = int(config["image_height"])
    caption_height = int(config["caption_height"])
    if min(cell_width, image_height, caption_height) <= 0:
        raise ValueError("Collage dimensions must be positive")
    font_size = int(config.get("font_size", 13))
    font = load_font(config.get("font_path"), font_size)
    value = Image.new("RGB", (cell_width, caption_height + image_height), "white")
    draw = ImageDraw.Draw(value)
    line_height = max(14, font_size + 3)
    for line_index, line in enumerate(lines):
        draw.text((7, 5 + line_index * line_height), str(line), fill="black", font=font)
    with Image.open(source) as opened:
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
    return (
        f"ACC: {float(row['ACC']):.3f}, NED: {float(row['NED']):.3f}, "
        f"CER: {float(row['CER']):.3f}"
    )


def result_lines(row: Mapping[str, Any], model_label: str) -> list[str]:
    return [model_label, f"target: {row['target_text']}", metric_line(row)]


def save_grid(grid: Sequence[Sequence[Image.Image]], destination: str | Path) -> Path:
    if not grid or not grid[0]:
        raise ValueError("A collage grid must not be empty")
    column_count = len(grid[0])
    if any(len(row) != column_count for row in grid):
        raise ValueError("Every collage row must have the same number of columns")
    tile_width, tile_height = grid[0][0].size
    if any(value.size != (tile_width, tile_height) for row in grid for value in row):
        raise ValueError("Every collage tile must have the same dimensions")
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
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    indexed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["sample_index"]), str(row["target_key"]))
        if key in indexed:
            raise ValueError(f"Duplicate result row for {key}")
        indexed[key] = row
    return indexed


def load_previous_sd35_rows(
    detailed_results: Path, previous_results_dir: Path
) -> list[dict[str, Any]]:
    """Load v1 metrics and repair the stale pre-rename generated-image paths."""

    if not detailed_results.is_file():
        raise FileNotFoundError(f"Missing version-1 detailed results: {detailed_results}")
    with detailed_results.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    selected: list[dict[str, Any]] = []
    for row in all_rows:
        if row.get("model") != "self_prompting_sd35":
            continue
        try:
            sample_index = int(row["sample_index"])
            target_key = str(row["target_key"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed version-1 detailed result row") from exc
        repaired = (
            previous_results_dir
            / "generated"
            / "self_prompting_sd35"
            / target_key
            / f"{sample_index:04d}.png"
        ).resolve()
        value = dict(row)
        value["sample_index"] = sample_index
        value["output_path"] = str(repaired)
        selected.append(value)
    if not selected:
        raise ValueError("No Self-Prompting SD3.5 v1 rows were found")
    return selected


def _first_samples(
    samples: Sequence[Mapping[str, Any]], count: int
) -> list[Mapping[str, Any]]:
    ordered = sorted(samples, key=lambda row: int(row["sample_index"]))
    if len(ordered) < count:
        raise ValueError(f"Collage requires at least {count} samples")
    return ordered[:count]


def render_case_collage(
    samples: Sequence[Mapping[str, Any]],
    version2_results: Sequence[Mapping[str, Any]],
    version1_results: Sequence[Mapping[str, Any]],
    destination: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """Render five rows: noisy source, v1 uppercase, and v2 lowercase.

    This asymmetric comparison intentionally follows the supplied v2
    specification literally.
    """

    current = index_results(version2_results)
    previous = index_results(version1_results)
    grid: list[list[Image.Image]] = []
    for sample in _first_samples(samples, 5):
        sample_index = int(sample["sample_index"])
        upper_key = (sample_index, "case_upper")
        lower_key = (sample_index, "case_lower")
        if upper_key not in previous:
            raise ValueError(f"Missing v1 uppercase collage result for sample {sample_index}")
        if lower_key not in current:
            raise ValueError(f"Missing v2 lowercase collage result for sample {sample_index}")
        grid.append(
            [
                tile(
                    sample["input_path"],
                    ["Noisy source", f"source: {sample['source_text']}"],
                    config,
                ),
                tile(
                    previous[upper_key]["output_path"],
                    result_lines(previous[upper_key], "Self Prompting SD3.5 v1"),
                    config,
                ),
                tile(
                    current[lower_key]["output_path"],
                    result_lines(current[lower_key], "Self Prompting SD3.5 v2"),
                    config,
                ),
            ]
        )
    return save_grid(grid, destination)


def render_special_collage(
    samples: Sequence[Mapping[str, Any]],
    version2_results: Sequence[Mapping[str, Any]],
    destination: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """Render the first noisy sample plus its six v2 special targets."""

    sample = _first_samples(samples, 1)[0]
    sample_index = int(sample["sample_index"])
    current = index_results(version2_results)
    row = [
        tile(
            sample["input_path"],
            ["Noisy source", f"source: {sample['source_text']}"],
            config,
        )
    ]
    for target_key in SPECIAL_TARGET_KEYS:
        key = (sample_index, target_key)
        if key not in current:
            raise ValueError(f"Missing v2 collage result for {key}")
        row.append(
            tile(
                current[key]["output_path"],
                result_lines(current[key], "Self Prompting SD3.5 v2"),
                config,
            )
        )
    return save_grid([row], destination)
