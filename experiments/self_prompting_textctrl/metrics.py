"""Text-recognition metrics used by the experiment."""
from __future__ import annotations
import math


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def text_metrics(target: str, prediction: str, *, case_sensitive: bool = True):
    if not case_sensitive:
        target, prediction = target.lower(), prediction.lower()
    distance = edit_distance(target, prediction)
    denominator = max(len(target), len(prediction), 1)
    return {"ACC": float(target == prediction), "NED": 1.0 - distance / denominator,
            "CER": distance / max(len(target), 1)}


def mean_std(values):
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
