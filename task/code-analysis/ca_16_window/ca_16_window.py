import math
from typing import List


def _require_values(values: List[float]) -> List[float]:
    if not isinstance(values, list):
        raise TypeError("values must be a list")
    for value in values:
        if not isinstance(value, (int, float)):
            raise TypeError("values must be numeric")
    return values


def _require_window(window: int) -> int:
    if not isinstance(window, int):
        raise TypeError("window must be an int")
    if window <= 0:
        raise ValueError("window must be positive")
    return window


def _average(chunk: List[float]) -> float:
    return math.floor((sum(chunk) / len(chunk)) * 100.0 + 0.5) / 100.0


def moving_average(values: List[float], window: int) -> List[float]:
    """Return rounded moving averages for each complete window.

    For a window of 3 over five values, return three averages: positions
    0..2, 1..3, and 2..4. If the window is larger than the input, return an
    empty list.
    """
    values = _require_values(values)
    window = _require_window(window)
    if window > len(values):
        return []
    result = []
    for start in range(0, len(values) - window):
        result.append(_average(values[start:start + window]))
    return result
