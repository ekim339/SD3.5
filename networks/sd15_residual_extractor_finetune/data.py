"""SRNet source/target pairs with training-identical canonical glyph rendering."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset


def read_labels(path):
    labels = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        if len(parts) != 2:
            raise ValueError("Expected filename and text at %s:%d" % (path, number))
        labels[parts[0]] = parts[1]
    return labels


class NeutralGlyphRenderer:
    """Exact implementation used by `encoders.dataset.NeutralGlyphRenderer`."""
    def __init__(self, font_path, canvas_size=(256, 256), padding=12):
        self.font_path = Path(font_path).expanduser().resolve()
        self.canvas_size = tuple(canvas_size)
        self.padding = int(padding)
        if not self.font_path.is_file():
            raise FileNotFoundError("Canonical font does not exist: %s" % self.font_path)

    def __call__(self, text):
        width, height = self.canvas_size
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        max_width, max_height = width - 2 * self.padding, height - 2 * self.padding
        size = max(8, height - 2 * self.padding)
        while size >= 8:
            font = ImageFont.truetype(str(self.font_path), size=size)
            box = draw.textbbox((0, 0), text, font=font)
            if box[2] - box[0] <= max_width and box[3] - box[1] <= max_height:
                break
            size -= 2
        if size < 8:
            font = ImageFont.truetype(str(self.font_path), size=8)
            box = draw.textbbox((0, 0), text, font=font)
        x = (width - (box[2] - box[0])) // 2 - box[0]
        y = (height - (box[3] - box[1])) // 2 - box[1]
        draw.text((x, y), text, font=font, fill="black")
        return image


def tensor_01(image):
    array = np.asarray(image, dtype="float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_residual(image):
    array = np.asarray(image, dtype="float32") / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def collect_records(roots):
    records = []
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        source_labels, target_labels = read_labels(root / "i_s.txt"), read_labels(root / "i_t.txt")
        if source_labels.keys() != target_labels.keys():
            raise ValueError("Source/target filenames differ under %s" % root)
        for filename in sorted(source_labels):
            source, target = root / "i_s" / filename, root / "t_f" / filename
            if not source.is_file() or not target.is_file():
                raise FileNotFoundError("Missing SRNet image pair for %s under %s" % (filename, root))
            if source_labels[filename] != target_labels[filename]:
                records.append((source, target, source_labels[filename], target_labels[filename]))
    if not records:
        raise ValueError("No same-style/different-text SRNet pairs found")
    return records


class SRNetResidualEditingDataset(Dataset):
    def __init__(self, records, canonical_font, resolution=256, condition_dropout=0.1):
        self.records = list(records)
        self.resolution = int(resolution)
        self.condition_dropout = float(condition_dropout)
        self.renderer = NeutralGlyphRenderer(canonical_font, (self.resolution, self.resolution))

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _open(path, size, resampling):
        with Image.open(path) as opened:
            return opened.convert("RGB").resize((size, size), resampling)

    def __getitem__(self, index):
        source_path, target_path, source_text, target_text = self.records[index]
        source_textctrl = self._open(source_path, self.resolution, Image.Resampling.BILINEAR)
        target_textctrl = self._open(target_path, self.resolution, Image.Resampling.BILINEAR)
        # Residual training used PIL LANCZOS followed by [-1,1] normalization.
        source_residual = self._open(source_path, self.resolution, Image.Resampling.LANCZOS)
        source_glyph = self.renderer(source_text)
        target_glyph = self.renderer(target_text)
        condition = "" if random.random() < self.condition_dropout else target_text
        return {
            "img": tensor_01(target_textctrl),
            "hint": tensor_01(source_textctrl),
            "source_residual": tensor_residual(source_residual),
            "source_glyph_residual": tensor_residual(source_glyph),
            "target_glyph": tensor_01(target_glyph),
            "texts": target_text,
            "cond": condition,
            "source_text": source_text,
        }


class SRNetResidualDataModule(pl.LightningDataModule):
    def __init__(self, data_config, **kwargs):
        super().__init__()
        self.config = data_config
        self.batch_size = int(data_config.batch_size)

    def setup(self, stage=None):
        records = collect_records(self.config.roots)
        limit = self.config.get("limit")
        if limit is not None:
            records = records[:int(limit)]
        count = int(self.config.validation_samples)
        if count <= 0 or count >= len(records):
            raise ValueError("validation_samples must be between 1 and dataset size - 1")
        generator = torch.Generator().manual_seed(int(self.config.split_seed))
        order = torch.randperm(len(records), generator=generator).tolist()
        validation_ids, training_ids = set(order[:count]), order[count:]
        validation = [record for index, record in enumerate(records) if index in validation_ids]
        training = [records[index] for index in training_ids]
        arguments = dict(canonical_font=self.config.canonical_font,
                         resolution=int(self.config.resolution),
                         condition_dropout=float(self.config.condition_dropout))
        self.train_data = SRNetResidualEditingDataset(training, **arguments)
        self.validation_data = SRNetResidualEditingDataset(validation, **arguments)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=int(self.config.num_workers), pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=int(self.config.num_workers), pin_memory=True,
                          drop_last=False)
