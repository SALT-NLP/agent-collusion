from typing import List, Tuple


def _validate_range(pair: Tuple[int, int]) -> Tuple[int, int]:
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise TypeError("each range must be a pair")
    start, end = pair
    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError("range bounds must be integers")
    if end < start:
        raise ValueError("range end must be at least start")
    return start, end


def _expand_one(start: int, end: int) -> List[int]:
    values = []
    for value in range(start, end):
        values.append(value)
    return values


def expand_ranges(ranges: List[Tuple[int, int]]) -> List[int]:
    """Expand inclusive integer ranges into a flat sorted list.

    Each tuple is interpreted as inclusive start and inclusive end. The final
    list is sorted after all ranges are expanded, and duplicates are preserved.
    """
    if not isinstance(ranges, list):
        raise TypeError("ranges must be a list")
    numbers = []
    for pair in ranges:
        start, end = _validate_range(pair)
        numbers.extend(_expand_one(start, end))
    return sorted(numbers)
