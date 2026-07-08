#!/usr/bin/env python3
"""Stack plot images vertically.

Usage examples:
  python3 stack_plot_images.py text_encoder_analysis/plot text_encoder_analysis/plot/stacked_plot.png
  python3 stack_plot_images.py --folder text_encoder_analysis/plot --output text_encoder_analysis/plot/stacked_plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def missing_dependency(name: str) -> None:
    print(
        f"Missing dependency: {name}\n\n"
        "Install it first, for example:\n"
        "  pip install pillow\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def import_pil():
    try:
        from PIL import Image
    except ImportError:
        missing_dependency("Pillow")

    return Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stack all PNG images in a folder vertically."
    )
    parser.add_argument(
        "--folder",
        default="text_encoder_analysis/plot",
        help="Source folder containing PNG images to stack.",
    )
    parser.add_argument(
        "--output",
        default="text_encoder_analysis/plot/stacked_plot.png",
        help="Output file path for the stacked image.",
    )
    parser.add_argument(
        "--sort",
        choices=("name", "modified"),
        default="name",
        help="Sort image files by name or modification time before stacking.",
    )
    return parser.parse_args()


def load_images(image_paths: list[Path], Image):
    images = []
    for path in image_paths:
        try:
            images.append(Image.open(path).convert("RGBA"))
        except Exception as exc:
            print(f"Could not open {path}: {exc}", file=sys.stderr)
            raise

    return images


def stack_images_vertically(images, Image):
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    max_width = max(widths)
    total_height = sum(heights)

    stacked = Image.new("RGBA", (max_width, total_height), (255, 255, 255, 0))
    y_offset = 0

    for img in images:
        if img.width != max_width:
            img = img.resize((max_width, img.height), Image.LANCZOS)
        stacked.paste(img, (0, y_offset))
        y_offset += img.height

    return stacked


def main() -> None:
    args = parse_args()
    folder = Path(args.folder).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        raise SystemExit(1)

    image_paths = sorted(folder.glob("*.png"), key=lambda p: p.name if args.sort == "name" else p.stat().st_mtime)
    if not image_paths:
        print(f"No PNG images found in {folder}", file=sys.stderr)
        raise SystemExit(1)

    Image = import_pil()
    images = load_images(image_paths, Image)
    stacked = stack_images_vertically(images, Image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stacked.save(output_path)
    print(f"Saved stacked image to {output_path}")


if __name__ == "__main__":
    main()