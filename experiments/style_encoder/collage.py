from __future__ import annotations

from pathlib import Path


def _font(path, size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _tile(image_path, lines, width, image_height, caption_height, font_path):
    from PIL import Image, ImageDraw
    tile = Image.new("RGB", (width, caption_height + image_height), "white")
    draw = ImageDraw.Draw(tile)
    font = _font(font_path, 12)
    for index, line in enumerate(lines):
        draw.text((6, 5 + index * 18), str(line), fill="black", font=font)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    image.thumbnail((width - 10, image_height - 10))
    tile.paste(image, ((width - image.width) // 2,
                       caption_height + (image_height - image.height) // 2))
    return tile


def _metrics(row):
    return f"ACC {row['ACC']:.0f} NED {row['NED']:.3f} CER {row['CER']:.3f}"


def render_collages(samples, rows, destination, config):
    from PIL import Image
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    width, image_height = int(config["cell_width"]), int(config["image_height"])
    caption, gap = int(config["caption_height"]), 8
    indexed = {(int(row["sample_index"]), row["experiment"],
                str(row["masked_square"]), float(row["masking_proportion"])): row for row in rows}
    by_id = {int(sample["sample_index"]): sample for sample in samples}

    # Required 5x7 layout: noisy sources, then the four non-zero proportion conditions.
    selected = [int(value) for value in config["proportion_sample_indices"]]
    canvas = Image.new("RGB", (7 * width, 5 * (caption + image_height)), (225, 225, 225))
    for column, sample_index in enumerate(selected):
        sample = by_id[sample_index]
        source = _tile(sample["input_path"],
                       [f"source {sample['source_text']}",
                        f"noise sigma {sample['input_noise_standard_deviation']:.3f}"],
                       width, image_height, caption, config["font_path"])
        canvas.paste(source, (column * width, 0))
        for row_index, proportion in enumerate((0.1, 0.3, 0.5, 0.7), 1):
            result = indexed[(sample_index, "proportion", "None", proportion)]
            tile = _tile(result["output_path"],
                         [f"target {result['target_text']} mask {proportion:g}",
                          f"OCR {result['ocr_predicted_text']}", _metrics(result)],
                         width, image_height, caption, config["font_path"])
            canvas.paste(tile, (column * width, row_index * (caption + image_height)))
    canvas.save(destination / "masking_proportions_5x7.png")

    # Three patch collages: a two-image header and the sixteen spatial interventions.
    for sample_index in [int(value) for value in config["patch_sample_indices"]]:
        sample = by_id[sample_index]
        patch_canvas = Image.new("RGB", (4 * width + 3 * gap,
                                         5 * (caption + image_height) + 4 * gap), (225, 225, 225))
        source = _tile(sample["input_path"],
                       [f"source {sample['source_text']}",
                        f"noise sigma {sample['input_noise_standard_deviation']:.3f}"],
                       width, image_height, caption, config["font_path"])
        baseline = indexed[(sample_index, "patch", "None", 0.0)]
        no_mask = _tile(baseline["output_path"],
                        [f"target {baseline['target_text']} no mask",
                         f"OCR {baseline['ocr_predicted_text']}", _metrics(baseline)],
                        width, image_height, caption, config["font_path"])
        patch_canvas.paste(source, (width + gap // 2, 0))
        patch_canvas.paste(no_mask, (2 * width + 3 * gap // 2, 0))
        for row in range(4):
            for col in range(4):
                result = indexed[(sample_index, "patch", str([row, col]), 0.0)]
                tile = _tile(result["output_path"],
                             [f"square [{row},{col}]", f"OCR {result['ocr_predicted_text']}",
                              _metrics(result)], width, image_height, caption, config["font_path"])
                patch_canvas.paste(tile, (col * (width + gap),
                                          (row + 1) * (caption + image_height + gap)))
        patch_canvas.save(destination / f"patches_sample_{sample_index:04d}.png")
