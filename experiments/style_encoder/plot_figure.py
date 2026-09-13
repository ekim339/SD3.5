"""Create separate OCR-metric and character-accuracy masking figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PACKAGE_DIR / "results" / "summary.csv"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "results" / "figures"

METRIC_SERIES = (
    ("ACC", "ACC_mean/std"),
    ("NED", "NED_mean/std"),
    ("CER", "CER_mean/std"),
)
CHARACTER_SERIES = tuple(
    (f"Char {position} accuracy", f"character_{position}_accuracy_mean/std")
    for position in range(1, 6)
)
SERIES = METRIC_SERIES + CHARACTER_SERIES
COLORS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9", "#6A3D9A", "#666666")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")


def parse_mean(value: str, *, column: str, line_number: int) -> float:
    try:
        return float(value.split("/", 1)[0])
    except (AttributeError, ValueError) as error:
        raise ValueError(
            f"Invalid {column!r} value at CSV line {line_number}: {value!r}"
        ) from error


def read_first_five(path: Path) -> list[dict[str, float]]:
    """Read exactly the requested numeric conditions 0.0 through 0.7."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"condition", *(column for _, column in SERIES)}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing summary columns: {', '.join(sorted(missing))}")
        rows = []
        for line_number, raw in enumerate(reader, 2):
            if len(rows) == 5:
                break
            try:
                condition = float(raw["condition"])
            except ValueError as error:
                raise ValueError(
                    f"The first five conditions must be numeric; line {line_number} is {raw['condition']!r}"
                ) from error
            row = {"condition": condition}
            for label, column in SERIES:
                row[label] = parse_mean(raw[column], column=column, line_number=line_number)
            rows.append(row)
    expected = [0.0, 0.1, 0.3, 0.5, 0.7]
    if [row["condition"] for row in rows] != expected:
        raise ValueError(f"Expected first five conditions {expected}")
    return rows


def plot_series(
    rows: list[dict[str, float]],
    series: tuple[tuple[str, str], ...],
    title: str,
    output_path: Path,
) -> Path:
    x_values = [row["condition"] for row in rows]
    figure, axis = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    for label, column in series:
        style_index = SERIES.index((label, column))
        axis.plot(
            x_values, [row[label] for row in rows], label=label,
            color=COLORS[style_index], marker=MARKERS[style_index],
            linewidth=2.0, markersize=6,
        )
    axis.set(title=title, xlabel="Masking proportion", ylabel="Metric value")
    axis.set_xticks(x_values)
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    axis.legend(frameon=False, ncols=min(len(series), 3), loc="best")
    axis.spines[["top", "right"]].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_figures(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    rows = read_first_five(input_path)
    return [
        plot_series(
            rows, METRIC_SERIES, "Style-Encoder Masking: OCR Metrics",
            output_dir / "style_masking_ocr_metrics.png",
        ),
        plot_series(
            rows, CHARACTER_SERIES,
            "Style-Encoder Masking: Character-Position Accuracy",
            output_dir / "style_masking_character_accuracy.png",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for output in generate_figures(args.input, args.output_dir):
        print(output)


if __name__ == "__main__":
    main()
