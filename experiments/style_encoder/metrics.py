from __future__ import annotations

import math


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def text_metrics(target: str, prediction: str, case_sensitive: bool = False):
    if not case_sensitive:
        target, prediction = target.casefold(), prediction.casefold()
    distance = levenshtein(target, prediction)
    return {"ACC": float(target == prediction),
            "NED": 1.0 - distance / max(len(target), len(prediction), 1),
            "CER": distance / max(len(target), 1)}


def position_scores(target: str, prediction: str, case_sensitive: bool = False):
    if not case_sensitive:
        target, prediction = target.casefold(), prediction.casefold()
    return [float(i < len(prediction) and prediction[i] == char)
            for i, char in enumerate(target)]


def mean_std(values):
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
