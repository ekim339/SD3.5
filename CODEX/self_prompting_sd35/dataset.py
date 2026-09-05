"""SRNet self-reconstruction and paired cooldown training data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class DatasetFormatError(ValueError):
    pass


TRAINING_MODES = ("self_reconstruction", "cooldown")


@dataclass(frozen=True)
class SRNetRecord:
    root: Path
    filename: str
    source_text: str
    target_text: str


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
        filename, text = parts
        if filename in result:
            raise DatasetFormatError(f"Duplicate filename {filename!r} in {path}")
        result[filename] = text
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
    *,
    target_image: Image.Image | None = None,
    target_mask: Image.Image | None = None,
    include_style_prompt: bool = True,
) -> dict[str, torch.Tensor]:
    """Build inputs, optionally using an aligned edited target for cooldown.

    ``mask`` is always the source/edit mask exposed to the model. A cooldown
    sample may additionally provide ``target_mask``; its union with the source
    mask is used only to weight the loss and never enters the transformer.
    Set ``include_style_prompt`` to false for self-reconstruction so the source
    text crop is neither constructed nor returned.
    """
    size = (resolution, resolution)
    source_canvas = fit_canvas(source.convert("RGB"), size, Image.Resampling.BICUBIC)
    target_canvas = (
        source_canvas
        if target_image is None
        else fit_canvas(target_image.convert("RGB"), size, Image.Resampling.BICUBIC)
    )
    mask_canvas = fit_canvas(mask.convert("L"), size, Image.Resampling.NEAREST).point(
        lambda p: 255 if p >= 128 else 0
    )
    glyph = render_glyph(text, size, font_path)
    source_tensor = TF.to_tensor(source_canvas)
    target_tensor = TF.to_tensor(target_canvas)
    mask_tensor = (TF.to_tensor(mask_canvas) >= 0.5).float()
    loss_mask = mask_tensor
    if target_mask is not None:
        target_mask_canvas = fit_canvas(
            target_mask.convert("L"), size, Image.Resampling.NEAREST
        ).point(lambda p: 255 if p >= 128 else 0)
        loss_mask = torch.maximum(
            mask_tensor, (TF.to_tensor(target_mask_canvas) >= 0.5).float()
        )
    sample = {
        "source_image": source_tensor.mul(2).sub(1),
        "target_image": target_tensor.mul(2).sub(1),
        "mask": mask_tensor,
        "loss_mask": loss_mask,
        "masked_image": (source_tensor * (1.0 - mask_tensor)).mul(2).sub(1),
        "glyph_image": TF.to_tensor(glyph).mul(2).sub(1),
    }
    if include_style_prompt:
        style = build_style_prompt(source, mask, size)
        sample["style_image"] = TF.to_tensor(style).mul(2).sub(1)
    return sample


class SRNetSelfPromptDataset(Dataset[dict[str, Any]]):
    """Build style-free self-reconstruction or style-conditioned edit samples."""

    def __init__(
        self,
        roots: Iterable[str | Path],
        resolution: int = 512,
        limit: int | None = 200_000,
        font_path: str | Path | None = None,
        mode: str = "self_reconstruction",
    ) -> None:
        if mode not in TRAINING_MODES:
            raise ValueError(
                f"Unsupported training mode {mode!r}; expected one of {TRAINING_MODES}"
            )
        self.resolution, self.font_path, self.mode = int(resolution), font_path, mode
        self.records: list[SRNetRecord] = []
        for root_value in roots:
            root = Path(root_value).expanduser().resolve()
            required_directories = ["i_s", "mask_s"]
            if mode == "cooldown":
                required_directories.extend(("t_f", "mask_t"))
            for directory in required_directories:
                if not (root / directory).is_dir():
                    raise DatasetFormatError(f"Missing directory: {root / directory}")
            source_labels = read_labels(root / "i_s.txt")
            target_labels = (
                read_labels(root / "i_t.txt") if mode == "cooldown" else source_labels
            )
            if source_labels.keys() != target_labels.keys():
                missing = sorted(source_labels.keys() - target_labels.keys())[:3]
                extra = sorted(target_labels.keys() - source_labels.keys())[:3]
                raise DatasetFormatError(
                    f"i_s.txt and i_t.txt filenames differ in {root}; "
                    f"missing targets={missing}, unexpected targets={extra}"
                )
            for name, source_text in sorted(source_labels.items()):
                target_text = target_labels[name]
                if mode == "cooldown" and source_text == target_text:
                    continue
                paths = [root / "i_s" / name, root / "mask_s" / name]
                if mode == "cooldown":
                    paths.extend((root / "t_f" / name, root / "mask_t" / name))
                if all(path.is_file() for path in paths):
                    self.records.append(
                        SRNetRecord(root, name, source_text, target_text)
                    )
        if limit is not None:
            self.records = self.records[: int(limit)]
        if not self.records:
            raise DatasetFormatError(f"No complete SRNet samples found for mode={mode!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.root / "i_s" / record.filename) as image:
            source = image.convert("RGB")
        with Image.open(record.root / "mask_s" / record.filename) as image:
            mask = image.convert("L")
        target = None
        target_mask = None
        if self.mode == "cooldown":
            with Image.open(record.root / "t_f" / record.filename) as image:
                target = image.convert("RGB")
            with Image.open(record.root / "mask_t" / record.filename) as image:
                target_mask = image.convert("L")
        sample: dict[str, Any] = prepare_conditions(
            source,
            mask,
            record.target_text,
            self.resolution,
            self.font_path,
            target_image=target,
            target_mask=target_mask,
            include_style_prompt=self.mode == "cooldown",
        )
        sample.update(
            filename=record.filename,
            source_text=record.source_text,
            target_text=record.target_text,
        )
        return sample
