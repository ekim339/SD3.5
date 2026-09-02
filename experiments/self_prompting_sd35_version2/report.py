"""Per-image metrics and baseline-preserving version-2 summary tables."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data import CASE_TARGET_KEYS, MODEL_KEY, SPECIAL_TARGET_KEYS, TARGET_KEYS
from .metrics import mean_std, text_metrics


METRICS = ("ACC", "NED", "CER")
MODEL_DISPLAY = "Self Prompting SD3.5 Version 2"
DETAILED_FIELDS = (
    "index",
    "sample_index",
    "filename",
    "model",
    "target_key",
    "source_text",
    "target_text",
    "ocr_predicted_text",
    *METRICS,
    "noise_standard_deviation",
    "noise_seed",
    "generation_seed",
    "source_path",
    "input_path",
    "mask_path",
    "output_path",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(
    jobs: Sequence[Mapping[str, Any]],
    predictions: Mapping[int, Mapping[str, Any]],
    *,
    case_sensitive: bool = True,
) -> list[dict[str, Any]]:
    """Join OCR predictions to jobs and compute requested per-image metrics."""

    expected_indices = {int(job["index"]) for job in jobs}
    if len(expected_indices) != len(jobs):
        raise ValueError("Job manifest contains duplicate indices")
    extra = sorted(set(predictions).difference(expected_indices))
    if extra:
        raise ValueError(f"OCR predictions contain unexpected job indices: {extra}")

    rows: list[dict[str, Any]] = []
    for job in jobs:
        index = int(job["index"])
        if index not in predictions:
            raise ValueError(f"Missing OCR prediction for job {index}")
        prediction_row = predictions[index]
        if "ocr_predicted_text" not in prediction_row:
            raise ValueError(f"OCR prediction {index} is missing ocr_predicted_text")
        expected_output = Path(str(job["output_path"])).resolve()
        if not expected_output.is_file():
            raise FileNotFoundError(
                f"Generated image for job {index} is missing: {expected_output}"
            )
        predicted_output = prediction_row.get("output_path")
        if predicted_output is not None and Path(str(predicted_output)).resolve() != expected_output:
            raise ValueError(f"OCR prediction {index} belongs to a different image")
        expected_digest = prediction_row.get("image_sha256")
        if expected_digest is not None and str(expected_digest) != _sha256_file(expected_output):
            raise ValueError(
                f"Generated image for OCR prediction {index} has changed"
            )
        if str(job.get("model")) != MODEL_KEY:
            raise ValueError(f"Job {index} has unexpected model={job.get('model')!r}")
        prediction = str(prediction_row["ocr_predicted_text"])
        rows.append(
            {
                **dict(job),
                "ocr_predicted_text": prediction,
                **text_metrics(
                    str(job["target_text"]),
                    prediction,
                    case_sensitive=case_sensitive,
                ),
            }
        )
    return rows


def validate_evaluated_rows(
    rows: Sequence[Mapping[str, Any]], expected_samples: int = 100
) -> None:
    expected_total = expected_samples * len(TARGET_KEYS)
    if len(rows) != expected_total:
        raise ValueError(f"Expected {expected_total} evaluated rows, found {len(rows)}")
    seen: set[tuple[int, str]] = set()
    counts = {target_key: 0 for target_key in TARGET_KEYS}
    for row in rows:
        if row.get("model") != MODEL_KEY:
            raise ValueError(f"Unexpected evaluated model: {row.get('model')!r}")
        key = (int(row["sample_index"]), str(row["target_key"]))
        if key in seen:
            raise ValueError(f"Duplicate evaluated sample/target pair: {key}")
        if key[1] not in counts:
            raise ValueError(f"Unexpected target key: {key[1]!r}")
        seen.add(key)
        counts[key[1]] += 1
    wrong = {key: count for key, count in counts.items() if count != expected_samples}
    if wrong:
        raise ValueError(f"Incorrect evaluated target counts: {wrong}")


def write_detailed(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> Path:
    destination = output_dir / "detailed_results.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAILED_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in DETAILED_FIELDS} for row in rows
        )
    return destination


def _summary_cells(
    rows: Sequence[Mapping[str, Any]], target_key: str
) -> list[str]:
    selected = [row for row in rows if row["target_key"] == target_key]
    if not selected:
        raise ValueError(f"No evaluated rows for target_key={target_key!r}")
    cells: list[str] = []
    for metric in METRICS:
        mean, standard_deviation = mean_std(float(row[metric]) for row in selected)
        cells.extend((f"{mean:.6f}", f"{standard_deviation:.6f}"))
    return cells


def _read_csv(path: Path, description: str) -> list[list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{description.capitalize()} is empty: {path}")
    return rows


def _write_csv(path: Path, rows: Sequence[Sequence[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return path


def write_case_summary(
    rows: Sequence[Mapping[str, Any]],
    previous_summary: Path,
    output_dir: Path,
) -> Path:
    """Copy the two v1 case tables and append one v2 row to each."""

    table = _read_csv(previous_summary, "version-1 capitalization summary")
    header = [
        "model",
        *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")),
    ]
    header_indices = [index for index, row in enumerate(table) if row == header]
    if len(header_indices) != 2:
        raise ValueError(
            f"Expected two capitalization tables in {previous_summary}; "
            f"found {len(header_indices)}"
        )
    labels = ("Capitalized target text", "Lowercase target text")
    for table_position in range(1, -1, -1):
        header_index = header_indices[table_position]
        expected_label = labels[table_position]
        if header_index == 0 or table[header_index - 1] != [expected_label]:
            raise ValueError(
                f"Expected {expected_label!r} immediately before table header in "
                f"{previous_summary}"
            )
        end = header_index + 1
        while end < len(table) and table[end]:
            end += 1
        baseline = [
            row for row in table[header_index + 1 : end]
            if not row or row[0] != MODEL_DISPLAY
        ]
        if not baseline:
            raise ValueError(f"No baseline rows in {expected_label!r}")
        if any(len(row) != len(header) for row in baseline):
            raise ValueError(f"Malformed baseline row in {expected_label!r}")
        new_row = [MODEL_DISPLAY, *_summary_cells(rows, CASE_TARGET_KEYS[table_position])]
        table[header_index + 1 : end] = [*baseline, new_row]
    return _write_csv(output_dir / "capital_lowercase_summary.csv", table)


def write_special_summary(
    rows: Sequence[Mapping[str, Any]],
    previous_summary: Path,
    output_dir: Path,
) -> Path:
    """Copy the v1 special table and append the 18-cell v2 metric row."""

    table = _read_csv(previous_summary, "version-1 special-character summary")
    expected_header = [
        "model",
        *(
            f"target_{target_index}_{metric}_mean/std"
            for target_index in range(1, 7)
            for metric in METRICS
        ),
    ]
    if table[0] != expected_header:
        raise ValueError(f"Unexpected special-character summary schema: {previous_summary}")
    baseline = [row for row in table[1:] if row and row[0] != MODEL_DISPLAY]
    if not baseline:
        raise ValueError(f"No baseline rows in {previous_summary}")
    if any(len(row) != len(expected_header) for row in baseline):
        raise ValueError(f"Malformed baseline row in {previous_summary}")
    values: list[str] = []
    for target_key in SPECIAL_TARGET_KEYS:
        selected = [row for row in rows if row["target_key"] == target_key]
        if not selected:
            raise ValueError(f"No evaluated rows for target_key={target_key!r}")
        for metric in METRICS:
            mean, standard_deviation = mean_std(float(row[metric]) for row in selected)
            values.append(f"{mean:.6f}/{standard_deviation:.6f}")
    table = [expected_header, *baseline, [MODEL_DISPLAY, *values]]
    return _write_csv(output_dir / "special_character_summary.csv", table)


def write_reports(
    rows: Sequence[Mapping[str, Any]],
    previous_case_summary: Path,
    previous_special_summary: Path,
    output_dir: Path,
    *,
    expected_samples: int = 100,
) -> list[Path]:
    validate_evaluated_rows(rows, expected_samples=expected_samples)
    return [
        write_detailed(rows, output_dir),
        write_case_summary(rows, previous_case_summary, output_dir),
        write_special_summary(rows, previous_special_summary, output_dir),
    ]
