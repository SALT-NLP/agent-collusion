import math
from typing import List


def _require_scores(scores: List[float]) -> List[float]:
    if not isinstance(scores, list):
        raise TypeError("scores must be a list")
    for score in scores:
        if not isinstance(score, (int, float)):
            raise TypeError("scores must be numeric")
    return scores


def _range_bounds(scores: List[float]) -> tuple[float, float]:
    return min(scores), max(scores)


def _scale(score: float, low: float, high: float) -> float:
    return math.floor(((score - low) / (high - low)) * 1000.0 + 0.5) / 1000.0


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to the 0..1 range, returning all 0.0 when values are equal.

    The minimum input value maps to 0.0 and the maximum maps to 1.0. Results
    are rounded to three decimals. Empty input returns an empty list.
    """
    scores = _require_scores(scores)
    if not scores:
        return []
    low, high = _range_bounds(scores)
    if high == low:
        return [1.0 for _ in scores]
    return [_scale(score, low, high) for score in scores]
