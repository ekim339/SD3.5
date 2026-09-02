"""Reuse the exact Version-1 samples and targets for the Version-2 experiment.

The Version-1 result manifests were moved after they were written, so their
absolute ``input_path`` values still name the former experiment directory.  A
Version-2 run must therefore locate the noisy image beside the supplied prior
manifest and copy its bytes into its own result tree.  Nothing is resampled or
re-noised here.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image


MODEL_KEY = "self_prompting_sd35_version2"
CASE_TARGET_KEYS = ("case_upper", "case_lower")
SPECIAL_TARGET_KEYS = tuple(f"special_{index}" for index in range(1, 7))
TARGET_KEYS = CASE_TARGET_KEYS + SPECIAL_TARGET_KEYS
EXPECTED_SAMPLE_COUNT = 100


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file and require every nonblank row to be an object."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON at {source}:{line_number}: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {source}:{line_number}")
        rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write a deterministic JSON-lines manifest and return its path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return destination


def _required_string(
    row: Mapping[str, Any], key: str, *, context: str
) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: {key!r} must be a non-empty string")
    return value


def _integer(value: Any, *, field: str, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}: {field!r} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: {field!r} must be an integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{context}: {field!r} must be an integer")
    if isinstance(value, str) and value.strip() != str(result):
        raise ValueError(f"{context}: {field!r} must be an integer")
    return result


def _float(value: Any, *, field: str, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context}: {field!r} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: {field!r} must be numeric") from error


def _validate_image(path: Path, *, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:
        raise ValueError(f"Unreadable {description}: {path}") from error


def _same_file_contents(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _locate_prior_noisy_input(
    sample: Mapping[str, Any], prior_manifest: Path
) -> Path:
    """Resolve a noisy input, rebasing a stale Version-1 absolute path."""

    context = f"sample {sample.get('sample_index', '?')}"
    stored = Path(_required_string(sample, "input_path", context=context)).expanduser()
    candidates = [stored]
    # The manifest now lives at ``<v1>/results/samples.jsonl`` and the exact
    # noisy images moved with it to ``<v1>/results/inputs``.
    candidates.append(prior_manifest.parent / "inputs" / stored.name)

    sample_index = _integer(
        sample.get("sample_index"), field="sample_index", context=context
    )
    filename = Path(_required_string(sample, "filename", context=context)).name
    candidates.append(
        prior_manifest.parent / "inputs" / f"{sample_index:04d}_{filename}"
    )
    existing: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.is_file() and resolved not in existing:
            existing.append(resolved)
    if not existing:
        rendered = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not locate the prior noisy input for {context}; checked {rendered}"
        )
    first = existing[0]
    for alternate in existing[1:]:
        if not _same_file_contents(first, alternate):
            raise ValueError(
                f"Conflicting noisy inputs for {context}: {first} and {alternate}"
            )
    return first


def _copy_exact(source: Path, destination: Path, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"Noisy-input destination is not a file: {destination}")
        if _same_file_contents(source, destination):
            return
        if not overwrite:
            raise FileExistsError(
                f"Existing noisy input differs from Version 1: {destination}; "
                "enable overwrite to replace it"
            )
    shutil.copy2(source, destination)
    if not _same_file_contents(source, destination):
        raise OSError(f"Copied noisy input differs from its source: {destination}")


def validate_samples(
    samples: Sequence[Mapping[str, Any]],
    expected_count: int = EXPECTED_SAMPLE_COUNT,
) -> None:
    """Validate the normalized sample manifest used by Version 2."""

    if len(samples) != expected_count:
        raise ValueError(f"Expected {expected_count} samples, found {len(samples)}")
    required = {
        "sample_index",
        "filename",
        "source_text",
        "source_path",
        "input_path",
        "mask_path",
        "noise_seed",
        "noise_standard_deviation",
        *TARGET_KEYS,
    }
    seen: set[int] = set()
    for row_number, sample in enumerate(samples, 1):
        context = f"sample row {row_number}"
        missing = sorted(required - sample.keys())
        if missing:
            raise ValueError(f"{context} is missing: {', '.join(missing)}")
        sample_index = _integer(
            sample["sample_index"], field="sample_index", context=context
        )
        if sample_index in seen:
            raise ValueError(f"Duplicate sample_index: {sample_index}")
        seen.add(sample_index)
        filename = _required_string(sample, "filename", context=context)
        if Path(filename).name != filename:
            raise ValueError(f"{context}: filename must be a basename: {filename!r}")
        source_text = _required_string(sample, "source_text", context=context)
        if len(source_text) != 5:
            raise ValueError(f"{context}: source_text must contain five characters")
        for target_key in TARGET_KEYS:
            _required_string(sample, target_key, context=context)
        if str(sample["case_upper"]) != str(sample["case_lower"]).upper():
            raise ValueError(
                f"{context}: case_upper is not the uppercase form of case_lower"
            )
        _integer(sample["noise_seed"], field="noise_seed", context=context)
        if _float(
            sample["noise_standard_deviation"],
            field="noise_standard_deviation",
            context=context,
        ) < 0:
            raise ValueError(f"{context}: noise_standard_deviation must be nonnegative")
        for key, description in (
            ("source_path", "source image"),
            ("input_path", "noise-perturbed input"),
            ("mask_path", "source mask"),
        ):
            value = Path(_required_string(sample, key, context=context))
            _validate_image(value, description=f"{description} for sample {sample_index}")
    expected_indices = set(range(expected_count))
    if seen != expected_indices:
        missing = sorted(expected_indices - seen)
        extra = sorted(seen - expected_indices)
        raise ValueError(
            f"sample_index values must be 0..{expected_count - 1}; "
            f"missing={missing}, extra={extra}"
        )


def prepare_samples(
    previous_samples_path: str | Path,
    output_dir: str | Path,
    *,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Copy the exact prior noisy inputs and write a relocated sample manifest."""

    prior_manifest = Path(previous_samples_path).expanduser().resolve(strict=False)
    prior_rows = read_jsonl(prior_manifest)
    if len(prior_rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} rows in {prior_manifest}, found {len(prior_rows)}"
        )
    destination_root = Path(output_dir).expanduser().resolve(strict=False)
    normalized: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for row_number, raw_sample in enumerate(prior_rows, 1):
        sample = dict(raw_sample)
        context = f"{prior_manifest}:{row_number}"
        sample_index = _integer(
            sample.get("sample_index"), field="sample_index", context=context
        )
        if sample_index in seen_indices:
            raise ValueError(f"Duplicate sample_index in prior manifest: {sample_index}")
        seen_indices.add(sample_index)
        filename = _required_string(sample, "filename", context=context)
        if Path(filename).name != filename:
            raise ValueError(f"{context}: filename must be a basename: {filename!r}")
        prior_input = _locate_prior_noisy_input(sample, prior_manifest)
        destination = (
            destination_root / "inputs" / f"{sample_index:04d}_{filename}"
        ).resolve(strict=False)
        _copy_exact(prior_input, destination, overwrite=overwrite)
        sample["sample_index"] = sample_index
        sample["noise_seed"] = _integer(
            sample.get("noise_seed"), field="noise_seed", context=context
        )
        sample["noise_standard_deviation"] = _float(
            sample.get("noise_standard_deviation"),
            field="noise_standard_deviation",
            context=context,
        )
        sample["source_path"] = str(
            Path(_required_string(sample, "source_path", context=context))
            .expanduser()
            .resolve(strict=False)
        )
        sample["mask_path"] = str(
            Path(_required_string(sample, "mask_path", context=context))
            .expanduser()
            .resolve(strict=False)
        )
        sample["prior_input_path"] = str(prior_input)
        sample["input_path"] = str(destination)
        normalized.append(sample)

    normalized.sort(key=lambda row: int(row["sample_index"]))
    validate_samples(normalized, expected_count)
    write_jsonl(destination_root / "samples.jsonl", normalized)
    return normalized


def _csv_value(row: Mapping[str, Any], key: str, *, context: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"{context}: missing {key!r}")
    return str(value)


def load_target_rows(
    previous_detailed_path: str | Path,
    samples: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
) -> list[dict[str, Any]]:
    """Deduplicate and validate the 800 prior sample/target combinations.

    Version 1 has one detailed row per model, so each sample/target pair occurs
    twice.  Duplicate model rows must agree on every piece of source, target,
    noise, and generation-seed metadata before one canonical row is retained.
    """

    validate_samples(samples, expected_count)
    sample_by_index = {int(row["sample_index"]): row for row in samples}
    source = Path(previous_detailed_path).expanduser().resolve(strict=False)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Detailed-results CSV has no header: {source}")
        required_columns = {
            "sample_index",
            "filename",
            "model",
            "target_key",
            "target_text",
            "source_text",
            "noise_standard_deviation",
            "noise_seed",
            "generation_seed",
        }
        missing = sorted(required_columns - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"Detailed-results CSV is missing columns: {', '.join(missing)}"
            )
        csv_rows = list(reader)
    if not csv_rows:
        raise ValueError(f"Detailed-results CSV is empty: {source}")

    grouped: dict[tuple[int, str], list[tuple[int, dict[str, str]]]] = {}
    for row_number, csv_row in enumerate(csv_rows, 2):
        context = f"{source}:{row_number}"
        sample_index = _integer(
            csv_row.get("sample_index"), field="sample_index", context=context
        )
        target_key = _csv_value(csv_row, "target_key", context=context)
        if target_key not in TARGET_KEYS:
            raise ValueError(f"{context}: unexpected target_key {target_key!r}")
        if sample_index not in sample_by_index:
            raise ValueError(f"{context}: unknown sample_index {sample_index}")
        grouped.setdefault((sample_index, target_key), []).append(
            (row_number, csv_row)
        )

    expected_keys = {
        (sample_index, target_key)
        for sample_index in range(expected_count)
        for target_key in TARGET_KEYS
    }
    actual_keys = set(grouped)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"Detailed-results target grid is incomplete; missing={missing}, extra={extra}"
        )

    canonical: list[dict[str, Any]] = []
    for sample_index in range(expected_count):
        sample = sample_by_index[sample_index]
        for target_key in TARGET_KEYS:
            duplicates = grouped[(sample_index, target_key)]
            model_rows = [
                _csv_value(row, "model", context=f"{source}:{row_number}")
                for row_number, row in duplicates
            ]
            expected_models = {"textctrl", "self_prompting_sd35"}
            if len(model_rows) != 2 or set(model_rows) != expected_models:
                raise ValueError(
                    "Expected exactly one TextCtrl and one Self-Prompting SD3.5 "
                    f"row for sample_index={sample_index}, target_key={target_key!r}; "
                    f"found models={model_rows}"
                )
            first_number, first = duplicates[0]
            first_context = f"{source}:{first_number}"
            target_text = _csv_value(first, "target_text", context=first_context)
            generation_seed = _integer(
                first.get("generation_seed"),
                field="generation_seed",
                context=first_context,
            )
            filename = _csv_value(first, "filename", context=first_context)
            source_text = _csv_value(first, "source_text", context=first_context)
            noise_seed = _integer(
                first.get("noise_seed"), field="noise_seed", context=first_context
            )
            noise_sigma = _float(
                first.get("noise_standard_deviation"),
                field="noise_standard_deviation",
                context=first_context,
            )
            invariant = (
                filename,
                source_text,
                target_text,
                noise_seed,
                noise_sigma,
                generation_seed,
            )
            for duplicate_number, duplicate in duplicates[1:]:
                duplicate_context = f"{source}:{duplicate_number}"
                duplicate_invariant = (
                    _csv_value(duplicate, "filename", context=duplicate_context),
                    _csv_value(duplicate, "source_text", context=duplicate_context),
                    _csv_value(duplicate, "target_text", context=duplicate_context),
                    _integer(
                        duplicate.get("noise_seed"),
                        field="noise_seed",
                        context=duplicate_context,
                    ),
                    _float(
                        duplicate.get("noise_standard_deviation"),
                        field="noise_standard_deviation",
                        context=duplicate_context,
                    ),
                    _integer(
                        duplicate.get("generation_seed"),
                        field="generation_seed",
                        context=duplicate_context,
                    ),
                )
                if duplicate_invariant != invariant:
                    raise ValueError(
                        f"Conflicting prior rows for sample_index={sample_index}, "
                        f"target_key={target_key!r}: lines {first_number} and "
                        f"{duplicate_number}"
                    )

            if filename != str(sample["filename"]):
                raise ValueError(
                    f"{first_context}: filename {filename!r} disagrees with samples.jsonl"
                )
            if source_text != str(sample["source_text"]):
                raise ValueError(
                    f"{first_context}: source_text disagrees with samples.jsonl"
                )
            if target_text != str(sample[target_key]):
                raise ValueError(
                    f"{first_context}: target_text disagrees with samples.jsonl"
                )
            if noise_seed != int(sample["noise_seed"]):
                raise ValueError(
                    f"{first_context}: noise_seed disagrees with samples.jsonl"
                )
            if noise_sigma != float(sample["noise_standard_deviation"]):
                raise ValueError(
                    f"{first_context}: noise_standard_deviation disagrees with samples.jsonl"
                )
            canonical.append(
                {
                    "sample_index": sample_index,
                    "filename": filename,
                    "target_key": target_key,
                    "target_text": target_text,
                    "generation_seed": generation_seed,
                }
            )

    expected_jobs = expected_count * len(TARGET_KEYS)
    if len(canonical) != expected_jobs:
        raise RuntimeError(f"Expected {expected_jobs} targets, built {len(canonical)}")
    return canonical


def expand_jobs(
    samples: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    model: str = MODEL_KEY,
) -> list[dict[str, Any]]:
    """Create the deterministic Version-2-only 800-job inference manifest."""

    if model != MODEL_KEY:
        raise ValueError(f"Version-2 jobs require model={MODEL_KEY!r}")
    sample_by_index = {int(row["sample_index"]): row for row in samples}
    expected_count = len(samples)
    validate_samples(samples, expected_count)
    expected_keys = {
        (sample_index, target_key)
        for sample_index in sample_by_index
        for target_key in TARGET_KEYS
    }
    indexed_targets: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row_number, target in enumerate(target_rows, 1):
        context = f"target row {row_number}"
        sample_index = _integer(
            target.get("sample_index"), field="sample_index", context=context
        )
        target_key = _required_string(target, "target_key", context=context)
        key = (sample_index, target_key)
        if key in indexed_targets:
            raise ValueError(f"Duplicate canonical target row for {key}")
        indexed_targets[key] = target
    if set(indexed_targets) != expected_keys:
        missing = sorted(expected_keys - set(indexed_targets))
        extra = sorted(set(indexed_targets) - expected_keys)
        raise ValueError(f"Target rows do not form a complete grid; missing={missing}, extra={extra}")

    destination_root = Path(output_dir).expanduser().resolve(strict=False)
    jobs: list[dict[str, Any]] = []
    for sample_index in sorted(sample_by_index):
        sample = sample_by_index[sample_index]
        for target_key in TARGET_KEYS:
            target = indexed_targets[(sample_index, target_key)]
            target_text = _required_string(target, "target_text", context=str((sample_index, target_key)))
            if target_text != str(sample[target_key]):
                raise ValueError(f"Target text for {(sample_index, target_key)} disagrees with sample")
            generation_seed = _integer(
                target.get("generation_seed"),
                field="generation_seed",
                context=str((sample_index, target_key)),
            )
            jobs.append(
                {
                    **dict(sample),
                    "index": len(jobs),
                    "model": MODEL_KEY,
                    "target_key": target_key,
                    "target_text": target_text,
                    "generation_seed": generation_seed,
                    "output_path": str(
                        (
                            destination_root
                            / "generated"
                            / MODEL_KEY
                            / target_key
                            / f"{sample_index:04d}.png"
                        ).resolve(strict=False)
                    ),
                }
            )
    return jobs

