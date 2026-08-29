"""SRNet self-reconstruction data and visual self-prompt construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class DatasetFormatError(ValueError):
    pass


def read_labels(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise DatasetFormatError(f"Missing label file: {path}")
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.strip().split(maxsplit=1)
        if not parts:
            continue
        if len(parts) != 2:
            raise DatasetFormatError(f"Expected `filename text` at {path}:{number}")
        result[parts[0]] = parts[1]
    return result


def fit_canvas(image: Image.Image, size: tuple[int, int], resample: Image.Resampling) -> Image.Image:
    image = image.copy()
    image.thumbnail(size, resample)
    canvas = Image.new(image.mode, size, 0)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def build_style_prompt(image: Image.Image, mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Crop the tight text rectangle and pad it to the model canvas."""
    binary = mask.convert("L").point(lambda p: 255 if p >= 128 else 0)
    box = binary.getbbox()
    crop = image.convert("RGB") if box is None else image.convert("RGB").crop(box)
    return fit_canvas(crop, size, Image.Resampling.BICUBIC)


def _load_font(path: str | Path | None, size: int):
    candidates = ([path] if path else []) + ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_glyph(text: str, size: tuple[int, int], font_path: str | Path | None = None) -> Image.Image:
    """Render a centered single-line white-on-black RGB glyph map."""
    gray = Image.new("L", size, 0)
    draw = ImageDraw.Draw(gray)
    font = _load_font(font_path, max(8, int(size[1] * 0.72)))
    while getattr(font, "size", 8) > 8:
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= size[0] * 0.94 and box[3] - box[1] <= size[1] * 0.9:
            break
        font = _load_font(font_path, font.size - 2)
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((size[0] - box[2] + box[0]) / 2 - box[0], (size[1] - box[3] + box[1]) / 2 - box[1]),
        text, fill=255, font=font,
    )
    return Image.merge("RGB", (gray, gray, gray))


def prepare_conditions(
    source: Image.Image,
    mask: Image.Image,
    text: str,
    resolution: int,
    font_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    size = (resolution, resolution)
    source_canvas = fit_canvas(source.convert("RGB"), size, Image.Resampling.BICUBIC)
    mask_canvas = fit_canvas(mask.convert("L"), size, Image.Resampling.NEAREST).point(
        lambda p: 255 if p >= 128 else 0
    )
    style = build_style_prompt(source, mask, size)
    glyph = render_glyph(text, size, font_path)
    source_tensor = TF.to_tensor(source_canvas)
    mask_tensor = (TF.to_tensor(mask_canvas) >= 0.5).float()
    return {
        "source_image": source_tensor.mul(2).sub(1),
        "target_image": source_tensor.mul(2).sub(1),
        "mask": mask_tensor,
        "masked_image": (source_tensor * (1.0 - mask_tensor)).mul(2).sub(1),
        "glyph_image": TF.to_tensor(glyph).mul(2).sub(1),
        "style_image": TF.to_tensor(style).mul(2).sub(1),
    }


class SRNetSelfPromptDataset(Dataset[dict[str, Any]]):
    """Use source image/text as both condition and reconstruction target."""

    def __init__(
        self,
        roots: Iterable[str | Path],
        resolution: int = 512,
        limit: int | None = 200_000,
        font_path: str | Path | None = None,
    ) -> None:
        self.resolution, self.font_path = int(resolution), font_path
        self.records: list[tuple[Path, str, str]] = []
        for root_value in roots:
            root = Path(root_value).expanduser().resolve()
            for directory in ("i_s", "mask_s"):
                if not (root / directory).is_dir():
                    raise DatasetFormatError(f"Missing directory: {root / directory}")
            for name, text in sorted(read_labels(root / "i_s.txt").items()):
                if (root / "i_s" / name).is_file() and (root / "mask_s" / name).is_file():
                    self.records.append((root, name, text))
        if limit is not None:
            self.records = self.records[: int(limit)]
        if not self.records:
            raise DatasetFormatError("No aligned SRNet source samples found")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        root, name, source_text = self.records[index]
        with Image.open(root / "i_s" / name) as image:
            source = image.convert("RGB")
        with Image.open(root / "mask_s" / name) as image:
            mask = image.convert("L")
        sample: dict[str, Any] = prepare_conditions(
            source, mask, source_text, self.resolution, self.font_path
        )
        sample.update(filename=name, source_text=source_text, target_text=source_text)
        return sample
