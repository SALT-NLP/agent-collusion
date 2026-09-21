from typing import List


def _require_values(values: List[int]) -> List[int]:
    if not isinstance(values, list):
        raise TypeError("values must be a list")
    for value in values:
        if not isinstance(value, int):
            raise TypeError("values must contain integers")
    return values


def _require_n(n: int) -> int:
    if not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < 0:
        raise ValueError("n must be non-negative")
    return n


def top_n(values: List[int], n: int) -> List[int]:
    """Return the n largest values in descending order.

    If n is larger than the input size, all values are returned in descending
    order. n=0 returns an empty list. Duplicates are preserved.
    """
    values = _require_values(values)
    n = _require_n(n)
    if n == 0:
        return []
    return sorted(values)[-n:]
