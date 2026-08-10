from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class DatasetFormatError(ValueError):
    pass


def _read_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not path.is_file():
        raise DatasetFormatError(f"Missing label file: {path}")
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise DatasetFormatError(f"Expected `filename text` at {path}:{line_number}")
        filename, text = parts
        if filename in labels:
            raise DatasetFormatError(f"Duplicate filename {filename!r} in {path}")
        labels[filename] = text.strip()
    return labels


class SRNetEditingDataset(Dataset[dict[str, Any]]):
    """Aligned SRNet samples for glyph/style-conditioned SD3.5 training."""

    required_directories = ("i_s", "mask_s", "t_f")

    def __init__(
        self,
        root: str | Path,
        resolution: int = 512,
        style_resolution: int = 256,
        prompt_template: str = 'Replace "{source_text}" with "{target_text}".',
        limit: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.resolution = resolution
        self.style_resolution = style_resolution
        self.prompt_template = prompt_template
        for directory in self.required_directories:
            if not (self.root / directory).is_dir():
                raise DatasetFormatError(f"Missing dataset directory: {self.root / directory}")
        source_labels = _read_labels(self.root / "i_s.txt")
        target_labels = _read_labels(self.root / "i_t.txt")
        if source_labels.keys() != target_labels.keys():
            raise DatasetFormatError("i_s.txt and i_t.txt must contain identical filenames")
        names = sorted(source_labels)
        if limit is not None:
            names = names[:limit]
        self.records = [(name, source_labels[name], target_labels[name]) for name in names]

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load_canvas(
        path: Path,
        resolution: int,
        mode: str,
        resample: Image.Resampling,
    ) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert(mode)
            scale = min(resolution / image.width, resolution / image.height)
            fitted = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            if fitted != image.size:
                image = image.resize(fitted, resample)
            canvas = Image.new(mode, (resolution, resolution), color=0)
            offset = (
                (resolution - image.width) // 2,
                (resolution - image.height) // 2,
            )
            canvas.paste(image, offset)
            return TF.to_tensor(canvas)

    @staticmethod
    def _load_rgb(path: Path, resolution: int) -> torch.Tensor:
        return SRNetEditingDataset._load_canvas(
            path, resolution, "RGB", Image.Resampling.BICUBIC
        )

    @staticmethod
    def _load_mask(path: Path, resolution: int) -> torch.Tensor:
        mask = SRNetEditingDataset._load_canvas(
            path, resolution, "L", Image.Resampling.NEAREST
        )
        return (mask >= 0.5).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename, source_text, target_text = self.records[index]
        source = self._load_rgb(self.root / "i_s" / filename, self.resolution)
        target = self._load_rgb(self.root / "t_f" / filename, self.resolution)
        mask = self._load_mask(self.root / "mask_s" / filename, self.resolution)
        style = self._load_rgb(self.root / "i_s" / filename, self.style_resolution)
        prompt = self.prompt_template.format(source_text=source_text, target_text=target_text)
        return {
            "filename": filename,
            "source_image": source.mul(2).sub(1),
            "target_image": target.mul(2).sub(1),
            "source_mask": mask,
            "style_image": style,
            "source_text": source_text,
            "target_text": target_text,
            "prompt": prompt,
        }
