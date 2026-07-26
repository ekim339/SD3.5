"""Validated loaders for JSONL and TextCtrl-style edit pairs."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scene_text_editing.configuration import resolve_path


class DatasetError(ValueError):
    """Raised when an editing dataset is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class EditingSample:
    sample_id: str
    source_image: Path
    source_text: str
    target_text: str
    mask_image: Path | None = None
    target_image: Path | None = None
    full_image: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TextImageEditingDataset(Sequence[EditingSample]):
    """Small immutable sequence suitable for inference and test fixtures."""

    def __init__(self, samples: Sequence[EditingSample]) -> None:
        self._samples = tuple(samples)

    def __getitem__(self, index: int) -> EditingSample:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[EditingSample]:
        return iter(self._samples)


def _resolve_record_path(root: Path, value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _require_file(
    path: Path | None,
    *,
    field_name: str,
    sample_id: str,
    strict: bool,
) -> None:
    if path is None:
        return
    if strict and not path.is_file():
        raise DatasetError(
            f"Sample {sample_id!r} references a missing {field_name}: {path}"
        )


def _validate_unique_samples(samples: Sequence[EditingSample]) -> None:
    seen: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen:
            raise DatasetError(f"Duplicate sample id: {sample.sample_id!r}")
        seen.add(sample.sample_id)


def _load_jsonl(config: Mapping[str, Any], root: Path) -> list[EditingSample]:
    manifest = _resolve_record_path(root, config.get("manifest", "manifest.jsonl"))
    if manifest is None or not manifest.is_file():
        raise DatasetError(f"JSONL manifest does not exist: {manifest}")
    strict = bool(config.get("strict", True))
    samples: list[EditingSample] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(
                    f"Invalid JSON at {manifest}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise DatasetError(
                    f"Expected an object at {manifest}:{line_number}."
                )
            missing = [
                key
                for key in ("source_image", "source_text", "target_text")
                if key not in record
            ]
            if missing:
                raise DatasetError(
                    f"Missing {', '.join(missing)} at {manifest}:{line_number}."
                )
            source_image = _resolve_record_path(root, record["source_image"])
            if source_image is None:
                raise DatasetError(
                    f"source_image must be non-empty at {manifest}:{line_number}."
                )
            sample_id = str(
                record.get("id") or record.get("sample_id") or source_image.stem
            )
            mask_image = _resolve_record_path(root, record.get("mask_image"))
            target_image = _resolve_record_path(root, record.get("target_image"))
            full_image = _resolve_record_path(root, record.get("full_image"))
            _require_file(
                source_image,
                field_name="source image",
                sample_id=sample_id,
                strict=strict,
            )
            _require_file(
                mask_image,
                field_name="mask image",
                sample_id=sample_id,
                strict=strict,
            )
            _require_file(
                target_image,
                field_name="target image",
                sample_id=sample_id,
                strict=strict,
            )
            _require_file(
                full_image,
                field_name="full image",
                sample_id=sample_id,
                strict=strict,
            )
            reserved = {
                "id",
                "sample_id",
                "source_image",
                "source_text",
                "target_text",
                "mask_image",
                "target_image",
                "full_image",
            }
            metadata = {
                key: value for key, value in record.items() if key not in reserved
            }
            samples.append(
                EditingSample(
                    sample_id=sample_id,
                    source_image=source_image,
                    source_text=str(record["source_text"]),
                    target_text=str(record["target_text"]),
                    mask_image=mask_image,
                    target_image=target_image,
                    full_image=full_image,
                    metadata=metadata,
                )
            )
    _validate_unique_samples(samples)
    if strict and not samples:
        raise DatasetError(f"JSONL manifest contains no samples: {manifest}")
    return samples


def _read_label_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise DatasetError(f"Label file does not exist: {path}")
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                raise DatasetError(
                    f"Expected `filename text` at {path}:{line_number}."
                )
            filename, label = parts[0], parts[1].strip()
            if filename in labels:
                raise DatasetError(
                    f"Duplicate filename {filename!r} in {path}:{line_number}."
                )
            labels[filename] = label
    return labels


def _load_textctrl(config: Mapping[str, Any], root: Path) -> list[EditingSample]:
    source_labels_path = _resolve_record_path(
        root, config.get("source_labels", "i_s.txt")
    )
    target_labels_path = _resolve_record_path(
        root, config.get("target_labels", "i_t.txt")
    )
    assert source_labels_path is not None
    assert target_labels_path is not None
    source_labels = _read_label_file(source_labels_path)
    target_labels = _read_label_file(target_labels_path)
    strict = bool(config.get("strict", True))

    source_names = set(source_labels)
    target_names = set(target_labels)
    if strict and source_names != target_names:
        only_source = sorted(source_names - target_names)
        only_target = sorted(target_names - source_names)
        raise DatasetError(
            "TextCtrl label files are not aligned by filename. "
            f"Only in source: {only_source}; only in target: {only_target}."
        )
    names = sorted(source_names & target_names)
    if strict and not names:
        raise DatasetError(f"No aligned TextCtrl samples found under {root}.")

    source_dir = _resolve_record_path(root, config.get("source_dir", "i_s"))
    target_dir = _resolve_record_path(root, config.get("target_dir"))
    mask_dir = _resolve_record_path(
        root,
        config.get("mask_dir", config.get("source_mask_dir")),
    )
    full_image_dir = _resolve_record_path(root, config.get("full_image_dir"))
    assert source_dir is not None
    samples: list[EditingSample] = []
    for filename in names:
        source_image = (source_dir / filename).resolve()
        target_image = (
            (target_dir / filename).resolve() if target_dir is not None else None
        )
        mask_image = (
            (mask_dir / filename).resolve() if mask_dir is not None else None
        )
        full_image = (
            (full_image_dir / filename).resolve()
            if full_image_dir is not None
            else None
        )
        sample_id = filename
        _require_file(
            source_image,
            field_name="source image",
            sample_id=sample_id,
            strict=strict,
        )
        _require_file(
            target_image,
            field_name="target image",
            sample_id=sample_id,
            strict=strict,
        )
        _require_file(
            mask_image,
            field_name="mask image",
            sample_id=sample_id,
            strict=strict,
        )
        _require_file(
            full_image,
            field_name="full image",
            sample_id=sample_id,
            strict=strict,
        )
        samples.append(
            EditingSample(
                sample_id=sample_id,
                source_image=source_image,
                source_text=source_labels[filename],
                target_text=target_labels[filename],
                mask_image=mask_image,
                target_image=target_image,
                full_image=full_image,
                metadata={"filename": filename},
            )
        )
    return samples


def _load_textctrl_shards(
    config: Mapping[str, Any],
    root: Path,
    *,
    split: str,
) -> list[EditingSample]:
    if split not in {"train", "validation"}:
        raise DatasetError("TextCtrl shard split must be train or validation.")
    pattern_key = "train_glob" if split == "train" else "validation_glob"
    pattern = str(config.get(pattern_key, f"{split}/*"))
    shard_dirs = sorted(path for path in root.glob(pattern) if path.is_dir())
    if bool(config.get("strict", True)) and not shard_dirs:
        raise DatasetError(
            f"No TextCtrl {split} shards match {root / pattern}."
        )
    samples: list[EditingSample] = []
    for shard in shard_dirs:
        shard_config = dict(config)
        shard_config["root_dir"] = str(shard)
        shard_config["format"] = "textctrl"
        shard_config["mask_dir"] = config.get("source_mask_dir")
        for sample in _load_textctrl(shard_config, shard):
            samples.append(
                EditingSample(
                    sample_id=f"{shard.name}/{sample.sample_id}",
                    source_image=sample.source_image,
                    source_text=sample.source_text,
                    target_text=sample.target_text,
                    mask_image=sample.mask_image,
                    target_image=sample.target_image,
                    full_image=sample.full_image,
                    metadata={**sample.metadata, "shard": shard.name},
                )
            )
    _validate_unique_samples(samples)
    return samples


def load_dataset(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    split: str = "train",
) -> TextImageEditingDataset:
    """Load one composed task dataset config."""

    root = resolve_path(str(config["root_dir"]), base_dir=base_dir)
    dataset_format = str(config.get("format", "")).lower()
    if dataset_format == "jsonl":
        samples = _load_jsonl(config, root)
    elif dataset_format == "textctrl":
        samples = _load_textctrl(config, root)
    elif dataset_format == "textctrl_shards":
        samples = _load_textctrl_shards(config, root, split=split)
    else:
        raise DatasetError(
            f"Unsupported text-image-editing dataset format: {dataset_format!r}."
        )
    return TextImageEditingDataset(samples)
