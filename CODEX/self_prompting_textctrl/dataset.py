from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class Sample:
    source: Path
    target: Path
    mask: Path
    target_text: str


def _read_labels(path: Path) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            filename, separator, text = line.partition(" ")
            if separator and filename and text:
                labels.append((filename, text))
    return labels


def discover_samples(
    root: str | Path, splits: Iterable[str], limit: int | None = None
) -> list[Sample]:
    root = Path(root)
    samples: list[Sample] = []
    for split_name in splits:
        split = root / split_name
        source_labels = split / "i_s.txt"
        target_labels = split / "i_t.txt"
        if not source_labels.is_file() or not target_labels.is_file():
            continue
        source_rows = _read_labels(source_labels)
        target_rows = _read_labels(target_labels)
        for (filename, _), (target_filename, target_text) in zip(
            source_rows, target_rows, strict=False
        ):
            if filename != target_filename:
                continue
            sample = Sample(
                source=split / "i_s" / filename,
                target=split / "t_f" / filename,
                mask=split / "mask_s" / filename,
                target_text=target_text,
            )
            # Dataset generation may still be running; skip assets not flushed yet.
            if sample.source.is_file() and sample.target.is_file() and sample.mask.is_file():
                samples.append(sample)
                if limit is not None and len(samples) >= limit:
                    return samples
    return samples


class GlyphRenderer:
    def __init__(self, font_path: str | Path, image_size: int) -> None:
        self.font_path = str(font_path)
        self.image_size = image_size

    def __call__(self, text: str) -> torch.Tensor:
        canvas = Image.new("L", (self.image_size, self.image_size), color=0)
        draw = ImageDraw.Draw(canvas)
        font = self._fit_font(draw, text)
        box = draw.textbbox((0, 0), text, font=font)
        width = max(box[2] - box[0], 1)
        height = max(box[3] - box[1], 1)
        position = (
            (self.image_size - width) // 2 - box[0],
            (self.image_size - height) // 2 - box[1],
        )
        draw.text(position, text, fill=255, font=font)
        return TF.to_tensor(canvas.convert("RGB"))

    def _fit_font(self, draw: ImageDraw.ImageDraw, text: str) -> ImageFont.FreeTypeFont:
        maximum_width = int(self.image_size * 0.9)
        maximum_height = int(self.image_size * 0.5)
        for size in range(maximum_height, 7, -2):
            font = ImageFont.truetype(self.font_path, size=size)
            box = draw.textbbox((0, 0), text, font=font)
            if box[2] - box[0] <= maximum_width and box[3] - box[1] <= maximum_height:
                return font
        return ImageFont.truetype(self.font_path, size=8)


class SRNetSelfPromptDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        root: str | Path,
        splits: Iterable[str],
        font_path: str | Path,
        image_size: int = 256,
        limit: int | None = 200_000,
    ) -> None:
        self.samples = discover_samples(root, splits, limit)
        if not self.samples:
            raise RuntimeError(f"No complete SRNet samples found under {root}")
        self.image_size = image_size
        self.render_glyph = GlyphRenderer(font_path, image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def _rgb(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = TF.resize(
            image, [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        )
        return TF.to_tensor(image)

    def _mask(self, path: Path) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        mask = TF.resize(mask, [self.image_size, self.image_size],
                         interpolation=InterpolationMode.NEAREST)
        return (TF.to_tensor(mask) >= 0.5).float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        source = self._rgb(sample.source)
        target = self._rgb(sample.target)
        mask = self._mask(sample.mask)
        glyph = self.render_glyph(sample.target_text)
        masked_source = source * (1.0 - mask)
        return {
            "source": source * 2.0 - 1.0,
            "target": target * 2.0 - 1.0,
            "masked_source": masked_source * 2.0 - 1.0,
            "style": source * 2.0 - 1.0,
            "glyph": glyph * 2.0 - 1.0,
            "mask": mask,
            "text": sample.target_text,
        }
