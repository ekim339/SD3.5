"""Render the requested capitalization and punctuation comparison collages."""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def _font(path, size=13):
    try: return ImageFont.truetype(str(path), size)
    except OSError: return ImageFont.load_default()


def _tile(path, lines, config):
    width, height = int(config["cell_width"]), int(config["image_height"])
    caption = int(config["caption_height"])
    tile = Image.new("RGB", (width, height + caption), "white")
    draw = ImageDraw.Draw(tile); font = _font(config["font_path"])
    for index, line in enumerate(lines[:4]):
        draw.text((6, 5 + 16 * index), str(line), fill="black", font=font)
    with Image.open(path) as opened: image = opened.convert("RGB")
    image.thumbnail((width - 8, height - 8))
    tile.paste(image, ((width-image.width)//2, caption+(height-image.height)//2))
    return tile


def _result_lines(row):
    return [f"target: {row['target_text']}", f"OCR: {row['ocr_predicted_text']}",
            f"ACC: {row['ACC']:.0f}, NED: {row['NED']:.3f}, CER: {row['CER']:.3f}"]


def _save_grid(rows, columns, destination):
    tile_width, tile_height = rows[0][0].size
    canvas = Image.new("RGB", (columns*tile_width, len(rows)*tile_height), (220,220,220))
    for row_index, row in enumerate(rows):
        for column_index, tile in enumerate(row):
            canvas.paste(tile, (column_index*tile_width, row_index*tile_height))
    destination.parent.mkdir(parents=True, exist_ok=True); canvas.save(destination)


def render_case_collage(samples, results, destination: Path, config):
    indexed = {(row["model"], int(row["sample_index"]), row["target_key"]): row for row in results}
    models = tuple(model for model in ("regular", "self_prompting")
                   if any(row["model"] == model for row in results))
    grid = []
    for sample in samples[:5]:
        index = int(sample["sample_index"])
        row = [_tile(sample["input_path"], [f"source: {sample['source_text']}"], config)]
        for model in models:
            for key in ("case_upper", "case_lower"):
                result = indexed[(model,index,key)]
                row.append(_tile(result["output_path"], _result_lines(result), config))
        grid.append(row)
    _save_grid(grid, 1 + 2 * len(models), destination)


def render_special_collage(samples, results, destination: Path, config):
    sample = samples[0]; index = int(sample["sample_index"])
    indexed = {(row["model"],int(row["sample_index"]),row["target_key"]):row for row in results}
    grid = []
    models = tuple(model for model in ("regular", "self_prompting")
                   if any(row["model"] == model for row in results))
    for model in models:
        row = [_tile(sample["input_path"], [f"{model}",f"source: {sample['source_text']}"], config)]
        for target_index in range(1,7):
            result = indexed[(model,index,f"special_{target_index}")]
            row.append(_tile(result["output_path"], _result_lines(result), config))
        grid.append(row)
    _save_grid(grid, 7, destination)
