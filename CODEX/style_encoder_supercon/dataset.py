"""SRNet style records and style-balanced minibatch construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class DatasetFormatError(ValueError):
    """Raised when an SRNet shard does not satisfy the required contract."""


def read_labels(path: Path) -> Dict[str, str]:
    if not path.is_file():
        raise DatasetFormatError("Missing metadata file: {}".format(path))
    values: Dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.strip().split(maxsplit=1)
        if not fields:
            continue
        if len(fields) != 2 or not fields[1].strip():
            raise DatasetFormatError(
                "Expected `filename text` at {}:{}".format(path, line_number)
            )
        filename, text = fields
        if filename in values:
            raise DatasetFormatError(
                "Duplicate filename {!r} in {}".format(filename, path)
            )
        values[filename] = text.strip()
    return values


@dataclass(frozen=True)
class StyleRecord:
    """One SRNet record with two strings in the same labeled font style."""

    style_id: int
    root: Path
    filename: str
    source_text: str
    target_text: str
    font: str

    @property
    def texts(self) -> Tuple[str, str]:
        return self.source_text, self.target_text


def _image_to_tensor(image) -> torch.Tensor:
    # TextCtrl trains its style encoder on RGB tensors in [0, 1].
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class SRNetStyleDataset(Dataset):
    """Return the `i_s`/`t_f` same-style pair from each SRNet record.

    The normalized font filename from ``font.txt`` is the style class (for
    example, Arial versus Times). SRNet's source/target images give every
    selected class two distinct-content positive views while also preserving
    the record's color/effects, layout, and background.
    """

    def __init__(
        self,
        roots: Sequence[Union[str, Path]],
        image_size: int = 128,
        limit: Optional[int] = None,
        require_different_text: bool = True,
        strict: bool = True,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive or null")
        self.image_size = int(image_size)
        records: List[StyleRecord] = []
        style_ids: Dict[str, int] = {}

        for raw_root in roots:
            root = Path(raw_root).expanduser().resolve()
            source = read_labels(root / "i_s.txt")
            target = read_labels(root / "i_t.txt")
            fonts = read_labels(root / "font.txt")
            if source.keys() != target.keys() or source.keys() != fonts.keys():
                raise DatasetFormatError("Metadata filenames differ in {}".format(root))
            for filename in sorted(source):
                if require_different_text and source[filename] == target[filename]:
                    continue
                missing = [
                    root / directory / filename
                    for directory in ("i_s", "t_f")
                    if not (root / directory / filename).is_file()
                ]
                if missing:
                    if strict:
                        raise DatasetFormatError("Missing paired image: {}".format(missing[0]))
                    continue
                font = fonts[filename]
                style_key = Path(font).name.casefold()
                records.append(
                    StyleRecord(
                        style_id=style_ids.setdefault(style_key, len(style_ids)),
                        root=root,
                        filename=filename,
                        source_text=source[filename],
                        target_text=target[filename],
                        font=font,
                    )
                )
                if limit is not None and len(records) >= limit:
                    break
            if limit is not None and len(records) >= limit:
                break
        if len(style_ids) < 2:
            raise DatasetFormatError("At least two font-style classes are required")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        from PIL import Image

        record = self.records[index]

        def load(directory: str) -> torch.Tensor:
            path = record.root / directory / record.filename
            with Image.open(path) as opened:
                image = opened.convert("RGB").resize(
                    (self.image_size, self.image_size), Image.Resampling.BICUBIC
                )
            return _image_to_tensor(image)

        return {
            "images": torch.stack((load("i_s"), load("t_f")), dim=0),
            "style_id": record.style_id,
            "texts": record.texts,
            "filename": record.filename,
            "font": record.font,
        }


class StyleBatchSampler(Sampler[List[int]]):
    """Sample distinct styles, preferring glyph-matched hard negatives.

    Each index becomes two same-font views in ``collate_style_pairs``. No font
    class is repeated within a batch. A hard negative is a different font class
    containing an exact text string already present in the batch.
    """

    def __init__(
        self,
        dataset: SRNetStyleDataset,
        indices: Sequence[int],
        styles_per_batch: int,
        hard_negative_probability: float = 0.5,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        if styles_per_batch < 2:
            raise ValueError("styles_per_batch must be at least 2")
        if not 0.0 <= hard_negative_probability <= 1.0:
            raise ValueError("hard_negative_probability must be in [0, 1]")
        unique = list(dict.fromkeys(int(index) for index in indices))
        classes = {dataset.records[index].style_id for index in unique}
        if len(classes) < styles_per_batch:
            raise ValueError("split has fewer font classes than styles_per_batch")
        self.dataset = dataset
        self.indices = unique
        self.styles_per_batch = int(styles_per_batch)
        self.hard_negative_probability = float(hard_negative_probability)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        by_text: Dict[str, Set[int]] = defaultdict(set)
        allowed = set(unique)
        for index in unique:
            for text in dataset.records[index].texts:
                by_text[text].add(index)
        self.by_text = {
            text: candidates & allowed for text, candidates in by_text.items()
        }

    def __len__(self) -> int:
        quotient, remainder = divmod(len(self.indices), self.styles_per_batch)
        return quotient if self.drop_last or remainder == 0 else quotient + 1

    def _hard_candidates(
        self, selected: Sequence[int], remaining: Set[int], selected_styles: Set[int]
    ) -> List[int]:
        candidates: Set[int] = set()
        for index in selected:
            for text in self.dataset.records[index].texts:
                candidates.update(self.by_text.get(text, set()))
        candidates.intersection_update(remaining)
        return sorted(
            index for index in candidates
            if self.dataset.records[index].style_id not in selected_styles
        )

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        by_style: Dict[int, List[int]] = defaultdict(list)
        for index in self.indices:
            by_style[self.dataset.records[index].style_id].append(index)
        if self.shuffle:
            for queue in by_style.values():
                rng.shuffle(queue)
        counts = {style: len(queue) for style, queue in by_style.items()}
        remaining = set(self.indices)

        def choose_style(styles: Sequence[int]) -> int:
            maximum = max(counts[style] for style in styles)
            tied = sorted(style for style in styles if counts[style] == maximum)
            return rng.choice(tied) if self.shuffle else tied[0]

        def take_from_style(style: int) -> int:
            queue = by_style[style]
            while queue and queue[-1] not in remaining:
                queue.pop()
            chosen = queue.pop()
            remaining.remove(chosen)
            counts[style] -= 1
            return chosen

        while remaining:
            available = [style for style, count in counts.items() if count > 0]
            if len(available) < self.styles_per_batch:
                break
            first_style = choose_style(available)
            first = take_from_style(first_style)
            batch = [first]
            selected_styles = {first_style}

            while len(batch) < self.styles_per_batch:
                hard = self._hard_candidates(batch, remaining, selected_styles)
                if hard and rng.random() < self.hard_negative_probability:
                    hard_styles = {self.dataset.records[index].style_id for index in hard}
                    chosen_style = choose_style(sorted(hard_styles))
                    matching = [
                        index for index in hard
                        if self.dataset.records[index].style_id == chosen_style
                    ]
                    chosen = rng.choice(matching) if self.shuffle else matching[0]
                    remaining.remove(chosen)
                    counts[chosen_style] -= 1
                else:
                    eligible_styles = [
                        style for style, count in counts.items()
                        if count > 0 and style not in selected_styles
                    ]
                    if not eligible_styles:
                        break
                    chosen_style = choose_style(eligible_styles)
                    chosen = take_from_style(chosen_style)
                batch.append(chosen)
                selected_styles.add(chosen_style)

            if len(batch) == self.styles_per_batch or (not self.drop_last and len(batch) >= 2):
                yield batch


def collate_style_pairs(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Flatten K style pairs into a 2K-image SupCon minibatch."""

    images = torch.stack([sample["images"] for sample in samples], dim=0)  # type: ignore[list-item]
    style_ids = torch.tensor([sample["style_id"] for sample in samples], dtype=torch.long)
    labels = style_ids[:, None].expand(-1, 2).reshape(-1)
    flattened_images = images.reshape(-1, *images.shape[2:])
    texts = [text for sample in samples for text in sample["texts"]]  # type: ignore[union-attr]
    return {
        "images": flattened_images,
        "style_labels": labels,
        "texts": texts,
        "filenames": [sample["filename"] for sample in samples],
        "fonts": [sample["font"] for sample in samples],
    }
