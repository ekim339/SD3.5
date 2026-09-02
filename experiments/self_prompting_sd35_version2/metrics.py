"""Case-sensitive OCR metrics for the Version-2 evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable


def edit_distance(left: str, right: str) -> int:
    """Return Levenshtein distance using O(min(len(left), len(right))) memory."""

    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("edit_distance inputs must be strings")
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def text_metrics(
    target: str, prediction: str, *, case_sensitive: bool = True
) -> dict[str, float]:
    """Compute exact-string ACC, normalized-edit similarity, and reference CER."""

    if not isinstance(target, str) or not isinstance(prediction, str):
        raise TypeError("target and prediction must be strings")
    if not case_sensitive:
        target, prediction = target.casefold(), prediction.casefold()
    distance = edit_distance(target, prediction)
    return {
        "ACC": float(target == prediction),
        "NED": 1.0 - distance / max(len(target), len(prediction), 1),
        "CER": distance / max(len(target), 1),
    }


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    """Return arithmetic mean and population standard deviation."""

    collected = [float(value) for value in values]
    if not collected:
        raise ValueError("Cannot summarize an empty metric collection")
    if any(not math.isfinite(value) for value in collected):
        raise ValueError("Metric values must be finite")
    mean = sum(collected) / len(collected)
    variance = sum((value - mean) ** 2 for value in collected) / len(collected)
    return mean, math.sqrt(variance)
