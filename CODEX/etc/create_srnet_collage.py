#!/usr/bin/env python3
"""Create a collage illustrating glyph and style pairings in SRNet_Datagen."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass(frozen=True)
class Sample:
    shard: Path
    filename: str
    source_text: str
    target_text: str
    font: str

    @property
    def source_image(self) -> Path:
        return self.shard / "i_s" / self.filename

    @property
    def styled_target_image(self) -> Path:
        return self.shard / "t_f" / self.filename


def read_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed metadata at {path}:{line_number}")
        values[fields[0]] = fields[1]
    return values


def load_samples(dataset_root: Path) -> list[Sample]:
    shards = sorted(dataset_root.glob("train/train-*"))
    if not shards:
        raise FileNotFoundError(f"No training shards found below {dataset_root}")

    samples: list[Sample] = []
    for shard in shards:
        source = read_metadata(shard / "i_s.txt")
        target = read_metadata(shard / "i_t.txt")
        fonts = read_metadata(shard / "font.txt")
        common = source.keys() & target.keys() & fonts.keys()
        for filename in sorted(common):
            sample = Sample(shard, filename, source[filename], target[filename], fonts[filename])
            if sample.source_image.is_file() and sample.styled_target_image.is_file():
                samples.append(sample)
    return samples


def select_pairs(samples: list[Sample], count: int) -> tuple[list[tuple[Sample, Sample]], list[Sample]]:
    """Choose exact same-text/different-font pairs and same-record style pairs."""
    by_glyph: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_glyph[sample.source_text].append(sample)

    glyph_pairs: list[tuple[Sample, Sample]] = []
    used: set[tuple[Path, str]] = set()
    for glyph in sorted(by_glyph, key=lambda text: (len(text), text.casefold())):
        candidates = by_glyph[glyph]
        first = candidates[0]
        second = next((item for item in candidates[1:] if item.font != first.font), None)
        if second is None:
            continue
        glyph_pairs.append((first, second))
        used.update(((first.shard, first.filename), (second.shard, second.filename)))
        if len(glyph_pairs) == count:
            break

    style_pairs = [
        sample
        for sample in samples
        if sample.source_text != sample.target_text
        and (sample.shard, sample.filename) not in used
    ][:count]
    if len(glyph_pairs) < count or len(style_pairs) < count:
        raise RuntimeError(
            f"Could only find {len(glyph_pairs)} same-glyph pairs and "
            f"{len(style_pairs)} same-style pairs; requested {count} of each"
        )
    return glyph_pairs, style_pairs


def display_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def image_tile(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.contain(source.convert("RGB"), size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", size, "white")
    tile.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return tile


def make_collage(
    glyph_pairs: list[tuple[Sample, Sample]], style_pairs: list[Sample], output: Path
) -> None:
    width, margin, gap = 1200, 36, 18
    tile_size = ((width - 2 * margin - gap) // 2, 180)
    title_font, label_font = display_font(32), display_font(19)
    section_h, label_h = 62, 48
    row_h = label_h + tile_size[1] + 20
    height = 2 * section_h + (len(glyph_pairs) + len(style_pairs)) * row_h + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0

    def section(title: str) -> None:
        nonlocal y
        draw.rectangle((0, y, width, y + section_h), fill="#17243a")
        draw.text((margin, y + 12), title, font=title_font, fill="white")
        y += section_h

    def row(left_path: Path, right_path: Path, left_label: str, right_label: str) -> None:
        nonlocal y
        x2 = margin + tile_size[0] + gap
        draw.text((margin, y + 8), left_label, font=label_font, fill="#202938")
        draw.text((x2, y + 8), right_label, font=label_font, fill="#202938")
        y += label_h
        canvas.paste(image_tile(left_path, tile_size), (margin, y))
        canvas.paste(image_tile(right_path, tile_size), (x2, y))
        draw.rectangle((margin, y, margin + tile_size[0], y + tile_size[1]), outline="#c4cad3", width=2)
        draw.rectangle((x2, y, x2 + tile_size[0], y + tile_size[1]), outline="#c4cad3", width=2)
        y += tile_size[1] + 20

    section("Same glyph, different style")
    for left, right in glyph_pairs:
        row(
            left.source_image,
            right.source_image,
            f'Glyph: "{left.source_text}"  |  Style: {Path(left.font).stem}',
            f'Glyph: "{right.source_text}"  |  Style: {Path(right.font).stem}',
        )

    section("Same style, different glyph")
    for sample in style_pairs:
        style = Path(sample.font).stem
        row(
            sample.source_image,
            sample.styled_target_image,
            f'Glyph: "{sample.source_text}"  |  Style: {style}',
            f'Glyph: "{sample.target_text}"  |  Style: {style}',
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/SRNet_Datagen"))
    parser.add_argument("--output", type=Path, default=Path("CODEX/etc/srnet_pairs_collage.png"))
    parser.add_argument("--pairs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pairs < 1:
        raise SystemExit("--pairs must be at least 1")
    samples = load_samples(args.dataset_root.expanduser().resolve())
    glyph_pairs, style_pairs = select_pairs(samples, args.pairs)
    output = args.output.expanduser().resolve()
    make_collage(glyph_pairs, style_pairs, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
