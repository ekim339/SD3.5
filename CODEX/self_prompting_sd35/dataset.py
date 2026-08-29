"""SRNet-Datagen input pipeline and self-prompt construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class DatasetFormatError(ValueError):
    """Raised when an SRNet shard is incomplete or unaligned."""


def read_labels(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise DatasetFormatError(f"Missing label file: {path}")
    labels: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        if len(parts) != 2:
            raise DatasetFormatError(f"Expected `filename text` at {path}:{line_number}")
        if parts[0] in labels:
            raise DatasetFormatError(f"Duplicate filename {parts[0]!r} in {path}")
        labels[parts[0]] = parts[1]
    return labels


def fit_canvas(image: Image.Image, size: tuple[int, int], resample: Image.Resampling) -> Image.Image:
    """Aspect-preserving resize centered on a black canvas."""
    image = image.copy()
    image.thumbnail(size, resample)
    canvas = Image.new(image.mode, size, 0)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def build_style_prompt(image: Image.Image, mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Crop the smallest masked rectangle and pad it back to the model canvas."""
    binary = mask.convert("L").point(lambda p: 255 if p >= 128 else 0)
    bbox = binary.getbbox()
    crop = image.convert("RGB") if bbox is None else image.convert("RGB").crop(bbox)
    return fit_canvas(crop, size, Image.Resampling.BICUBIC)


def _font(font_path: str | Path | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: Iterable[str | Path] = (() if font_path is None else (font_path,))
    candidates = (*candidates, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), font_size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_glyph(text: str, size: tuple[int, int], font_path: str | Path | None = None) -> Image.Image:
    """Render one centered, single-line white-on-black target glyph map."""
    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    # Find the largest font that fits without wrapping.
    font = _font(font_path, max(8, int(size[1] * 0.72)))
    while getattr(font, "size", 8) > 8:
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= size[0] * 0.94 and box[3] - box[1] <= size[1] * 0.9:
            break
        font = _font(font_path, font.size - 2)
    box = draw.textbbox((0, 0), text, font=font)
    x = (size[0] - (box[2] - box[0])) / 2 - box[0]
    y = (size[1] - (box[3] - box[1])) / 2 - box[1]
    draw.text((x, y), text, fill=255, font=font)
    return Image.merge("RGB", (canvas, canvas, canvas))


class SRNetSelfPromptDataset(Dataset[dict[str, Any]]):
    """Aligned source/target pairs from one or more SRNet 50k shards."""

    def __init__(
        self,
        roots: Iterable[str | Path],
        resolution: int = 512,
        limit: int | None = 200_000,
        font_path: str | Path | None = None,
    ) -> None:
        self.resolution = int(resolution)
        self.font_path = font_path
        self.records: list[tuple[Path, str, str, str]] = []
        for value in roots:
            root = Path(value).expanduser().resolve()
            for directory in ("i_s", "t_f", "mask_s"):
                if not (root / directory).is_dir():
                    raise DatasetFormatError(f"Missing directory: {root / directory}")
            sources, targets = read_labels(root / "i_s.txt"), read_labels(root / "i_t.txt")
            common = sorted(sources.keys() & targets.keys())
            for name in common:
                paths = (root / "i_s" / name, root / "t_f" / name, root / "mask_s" / name)
                if all(path.is_file() for path in paths):
                    self.records.append((root, name, sources[name], targets[name]))
        if limit is not None:
            self.records = self.records[: int(limit)]
        if not self.records:
            raise DatasetFormatError("No aligned SRNet samples found")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        root, name, source_text, target_text = self.records[index]
        size = (self.resolution, self.resolution)
        with Image.open(root / "i_s" / name) as opened:
            source_original = opened.convert("RGB")
        with Image.open(root / "t_f" / name) as opened:
            target = fit_canvas(opened.convert("RGB"), size, Image.Resampling.BICUBIC)
        with Image.open(root / "mask_s" / name) as opened:
            mask_original = opened.convert("L")
        source = fit_canvas(source_original, size, Image.Resampling.BICUBIC)
        mask = fit_canvas(mask_original, size, Image.Resampling.NEAREST).point(
            lambda p: 255 if p >= 128 else 0
        )
        style = build_style_prompt(source_original, mask_original, size)
        glyph = render_glyph(target_text, size, self.font_path)
        source_t, target_t = TF.to_tensor(source), TF.to_tensor(target)
        mask_t = (TF.to_tensor(mask) >= 0.5).float()
        return {
            "filename": name,
            "source_image": source_t.mul(2).sub(1),
            "target_image": target_t.mul(2).sub(1),
            "mask": mask_t,
            "masked_image": (source_t * (1.0 - mask_t)).mul(2).sub(1),
            "glyph_image": TF.to_tensor(glyph).mul(2).sub(1),
            "style_image": TF.to_tensor(style).mul(2).sub(1),
            "source_text": source_text,
            "target_text": target_text,
        }
