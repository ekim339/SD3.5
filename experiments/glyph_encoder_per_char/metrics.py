from __future__ import annotations
import math

def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + (x != y)))
        prev = cur
    return prev[-1]

def text_metrics(target, prediction, case_sensitive=False):
    if not case_sensitive: target, prediction = target.casefold(), prediction.casefold()
    distance = levenshtein(target, prediction)
    return {"ACC": float(target == prediction),
            "NED": 1 - distance / max(len(target), len(prediction), 1),
            "CER": distance / max(len(target), 1)}

def position_scores(target, prediction, case_sensitive=False):
    if not case_sensitive: target, prediction = target.casefold(), prediction.casefold()
    return [float(i < len(prediction) and prediction[i] == char)
            for i, char in enumerate(target)]

def mean_std(values):
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
