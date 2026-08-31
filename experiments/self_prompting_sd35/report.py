"""Detailed and aggregate CSV reporting for both requested evaluations."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data import MODEL_KEYS, SPECIAL_TARGET_KEYS
from .metrics import mean_std, text_metrics


METRICS = ("ACC", "NED", "CER")
MODEL_DISPLAY = {
    "textctrl": "TextCtrl",
    "self_prompting_sd35": "Self Prompting SD3.5",
}


def evaluate(
    jobs: Sequence[Mapping[str, Any]],
    predictions: Mapping[int, Mapping[str, Any]],
    *,
    case_sensitive: bool = True,
) -> list[dict[str, Any]]:
    """Join OCR predictions to jobs and compute per-image metrics."""

    rows: list[dict[str, Any]] = []
    for job in jobs:
        index = int(job["index"])
        if index not in predictions:
            raise ValueError(f"Missing OCR prediction for job {index}")
        prediction_row = predictions[index]
        expected_output = Path(str(job["output_path"])).resolve()
        predicted_output = prediction_row.get("output_path")
        if predicted_output and Path(str(predicted_output)).resolve() != expected_output:
            raise ValueError(f"OCR prediction {index} belongs to a different generated image")
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


def present_models(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    available = {str(row["model"]) for row in rows}
    return tuple(model for model in MODEL_KEYS if model in available)


def members(
    rows: Sequence[Mapping[str, Any]], model: str, target_key: str
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row["model"] == model and row["target_key"] == target_key
    ]
    if not selected:
        raise ValueError(f"No rows for model={model!r}, target_key={target_key!r}")
    return selected


def write_detailed(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    fields = [
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
    ]
    destination = output_dir / "detailed_results.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    return destination


def write_case_summary(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    """Write two vertically separated tables to the requested CSV."""

    destination = output_dir / "capital_lowercase_summary.csv"
    header = [
        "model",
        *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")),
    ]
    tables = (("Capitalized target text", "case_upper"), ("Lowercase target text", "case_lower"))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for table_index, (label, target_key) in enumerate(tables):
            if table_index:
                writer.writerow([])
            writer.writerow([label])
            writer.writerow(header)
            for model in present_models(rows):
                group = members(rows, model, target_key)
                values: list[str] = []
                for metric in METRICS:
                    mean, standard_deviation = mean_std(row[metric] for row in group)
                    values.extend((f"{mean:.6f}", f"{standard_deviation:.6f}"))
                writer.writerow([MODEL_DISPLAY[model], *values])
    return destination


def write_special_summary(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Path:
    """Write six target families x three mean/std metric cells."""

    destination = output_dir / "special_character_summary.csv"
    header = [
        "model",
        *(
            f"target_{target_index}_{metric}_mean/std"
            for target_index in range(1, 7)
            for metric in METRICS
        ),
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for model in present_models(rows):
            values: list[str] = []
            for target_key in SPECIAL_TARGET_KEYS:
                group = members(rows, model, target_key)
                for metric in METRICS:
                    mean, standard_deviation = mean_std(row[metric] for row in group)
                    values.append(f"{mean:.6f}/{standard_deviation:.6f}")
            writer.writerow([MODEL_DISPLAY[model], *values])
    return destination


def write_reports(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        write_detailed(rows, output_dir),
        write_case_summary(rows, output_dir),
        write_special_summary(rows, output_dir),
    ]

