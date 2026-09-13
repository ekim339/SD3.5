from pathlib import Path

from experiments.style_encoder.plot_figure import (
    CHARACTER_SERIES,
    METRIC_SERIES,
    read_first_five,
)


def test_reads_only_requested_masking_rows_and_splits_series():
    rows = read_first_five(Path("experiments/style_encoder/results/summary.csv"))
    assert [row["condition"] for row in rows] == [0.0, 0.1, 0.3, 0.5, 0.7]
    assert rows[0]["ACC"] == 0.22
    assert rows[-1]["Char 5 accuracy"] == 0.78
    assert [label for label, _ in METRIC_SERIES] == ["ACC", "NED", "CER"]
    assert [label for label, _ in CHARACTER_SERIES] == [
        "Char 1 accuracy", "Char 2 accuracy", "Char 3 accuracy",
        "Char 4 accuracy", "Char 5 accuracy",
    ]
