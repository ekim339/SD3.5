#!/usr/bin/env python3
"""Create 2x3 image collages for each seed across prompt subfolders in outputs/."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 2x3 collages from seed-matched images in outputs/ subfolders."
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing prompt subfolders with seed image files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/collages",
        help="Directory where collage images will be written.",
    )
    return parser.parse_args()


def import_pil():
    from PIL import Image, ImageDraw, ImageFont

    return Image, ImageDraw, ImageFont


def discover_prompt_folders(outputs_dir: Path) -> List[Path]:
    return sorted(
        [path for path in outputs_dir.iterdir() if path.is_dir() and not path.name.startswith(".")],
        key=lambda p: p.name,
    )


def discover_seeds(prompt_folders: List[Path]) -> List[int]:
    seeds = set()
    for folder in prompt_folders:
        for image_path in folder.glob("seed_*.png"):
            stem = image_path.stem
            if stem.startswith("seed_"):
                try:
                    seeds.add(int(stem.split("_", 1)[1]))
                except ValueError:
                    continue
    return sorted(seeds)


def build_label(folder_name: str) -> str:
    if folder_name.startswith("bathroom_"):
        return folder_name[len("bathroom_"):]
    return folder_name


def make_tile(Image, ImageDraw, ImageFont, image_path: Path, label: str, tile_size: Tuple[int, int]):
    img = Image.open(image_path).convert("RGBA")
    img = img.resize(tile_size, Image.LANCZOS)

    tile = Image.new("RGBA", tile_size, (255, 255, 255, 255))
    tile.paste(img, (0, 0))

    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    padding = 8
    text_bg = Image.new("RGBA", (text_width + padding * 2, text_height + padding * 2), (255, 255, 255, 180))
    tile.paste(text_bg, (padding, padding), text_bg)
    draw = ImageDraw.Draw(tile)
    draw.text((padding * 2, padding), label, fill=(0, 0, 0, 255), font=font)
    return tile


def create_collage(Image, ImageDraw, ImageFont, tiles: List[Image.Image], output_path: Path, cols: int = 3, rows: int = 2) -> None:
    tile_width, tile_height = tiles[0].size
    canvas_width = cols * tile_width + (cols - 1) * 16
    canvas_height = rows * tile_height + (rows - 1) * 16
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = col * (tile_width + 16)
        y = row * (tile_height + 16)
        canvas.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not outputs_dir.exists():
        raise SystemExit(f"Outputs directory not found: {outputs_dir}")

    Image, ImageDraw, ImageFont = import_pil()
    prompt_folders = discover_prompt_folders(outputs_dir)
    if not prompt_folders:
        raise SystemExit(f"No subfolders found under {outputs_dir}")

    seeds = discover_seeds(prompt_folders)
    if not seeds:
        raise SystemExit("No seed images were found")

    output_dir.mkdir(parents=True, exist_ok=True)
    tile_size = (1024, 1024)

    for seed in seeds:
        tiles = []
        for folder in prompt_folders:
            image_path = folder / f"seed_{seed}.png"
            if not image_path.exists():
                continue
            label = build_label(folder.name)
            tiles.append(make_tile(Image, ImageDraw, ImageFont, image_path, label, tile_size))

        if len(tiles) != len(prompt_folders):
            print(f"Skipping seed {seed}: missing images in some folders")
            continue

        collage_path = output_dir / f"collage_seed_{seed}.png"
        create_collage(Image, ImageDraw, ImageFont, tiles, collage_path)
        print(f"Saved {collage_path}")


if __name__ == "__main__":
    main()
