from __future__ import annotations

import csv
from collections import defaultdict

from .metrics import mean_std, position_scores, text_metrics


def evaluate(jobs, predictions, case_sensitive=False):
    rows = []
    for job in jobs:
        prediction = str(predictions[int(job["index"])]["ocr_predicted_text"])
        positions = position_scores(job["target_text"], prediction, case_sensitive)
        rows.append({**job, "ocr_predicted_text": prediction,
                     **text_metrics(job["target_text"], prediction, case_sensitive),
                     **{f"character_{i + 1}_correct": score for i, score in enumerate(positions)}})
    return rows


def _cell(rows, key):
    mean, std = mean_std([float(row[key]) for row in rows])
    return f"{mean:.6f}/{std:.6f}"


def write_reports(rows, output_dir):
    details = ["index", "experiment", "sample_index", "filename", "source_text", "target_text",
               "input_noise_standard_deviation", "input_noise_seed", "masking_proportion",
               "masked_square", "masked_token_indices", "style_mask_seed", "generation_seed",
               "ocr_predicted_text", "ACC", "NED", "CER",
               *[f"character_{i}_correct" for i in range(1, 6)], "input_path", "output_path"]
    with (output_dir / "detailed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=details)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in details} for row in rows)

    groups = defaultdict(list)
    for row in rows:
        key = ("proportion", float(row["masking_proportion"])) if row["experiment"] == "proportion" \
              else ("patch", str(row["masked_square"]))
        groups[key].append(row)
    header = ["condition", "ACC_mean/std", "NED_mean/std", "CER_mean/std",
              *[f"character_{i}_accuracy_mean/std" for i in range(1, 6)]]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for proportion in (0.0, 0.1, 0.3, 0.5, 0.7):
            members = groups[("proportion", proportion)]
            writer.writerow([proportion, *[_cell(members, key) for key in ("ACC", "NED", "CER")],
                             *[_cell(members, f"character_{i}_correct") for i in range(1, 6)]])
        for square in ["None"] + [str([row, col]) for row in range(4) for col in range(4)]:
            members = groups[("patch", square)]
            label = "patch baseline (no mask)" if square == "None" else square
            writer.writerow([label, *[_cell(members, key) for key in ("ACC", "NED", "CER")],
                             *[_cell(members, f"character_{i}_correct") for i in range(1, 6)]])
