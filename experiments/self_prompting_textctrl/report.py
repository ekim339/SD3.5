"""Detailed and aggregate CSV reports for both requested evaluations."""
from __future__ import annotations
import csv
from pathlib import Path
from .metrics import mean_std, text_metrics

METRICS = ("ACC", "NED", "CER")
MODEL_ORDER = ("regular", "self_prompting")


def present_models(rows):
    available = {row["model"] for row in rows}
    return tuple(model for model in MODEL_ORDER if model in available)


def evaluate(jobs, predictions, *, case_sensitive=True):
    rows = []
    for job in jobs:
        prediction = str(predictions[int(job["index"])]["ocr_predicted_text"])
        rows.append({**job, "ocr_predicted_text": prediction,
                     **text_metrics(str(job["target_text"]), prediction,
                                    case_sensitive=case_sensitive)})
    return rows


def _members(rows, model, target_key):
    return [row for row in rows if row["model"] == model and row["target_key"] == target_key]


def write_detailed(rows, output_dir: Path):
    fields = ["index", "sample_index", "filename", "model", "target_key", "source_text",
              "target_text", "ocr_predicted_text", *METRICS, "noise_sigma", "noise_seed",
              "generation_seed", "input_path", "mask_path", "output_path"]
    with (output_dir / "detailed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def write_case_summary(rows, output_dir: Path):
    path = output_dir / "capital_lowercase_summary.csv"
    header = ["model", *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std"))]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for position, (label, target_key) in enumerate((("capitalized", "case_upper"),
                                                        ("lowercase", "case_lower"))):
            if position: writer.writerow([])
            writer.writerow([label]); writer.writerow(header)
            for model in present_models(rows):
                members = _members(rows, model, target_key); cells = []
                for metric in METRICS: cells.extend(mean_std(row[metric] for row in members))
                writer.writerow([model, *[f"{value:.6f}" for value in cells]])


def write_special_summary(rows, output_dir: Path):
    target_keys = [f"special_{index}" for index in range(1, 7)]
    header = ["model", *(f"target_{index}_{metric}_mean/std"
              for index in range(1, 7) for metric in METRICS)]
    with (output_dir / "special_character_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(header)
        for model in present_models(rows):
            cells = []
            for target_key in target_keys:
                members = _members(rows, model, target_key)
                for metric in METRICS:
                    mean, std = mean_std(row[metric] for row in members)
                    cells.append(f"{mean:.6f}/{std:.6f}")
            writer.writerow([model, *cells])


def write_reports(rows, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_detailed(rows, output_dir); write_case_summary(rows, output_dir)
    write_special_summary(rows, output_dir)
