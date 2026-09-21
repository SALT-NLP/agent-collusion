from typing import List


def _require_int_list(name: str, values: List[int]) -> List[int]:
    if not isinstance(values, list):
        raise TypeError(f"{name} must be a list")
    for value in values:
        if not isinstance(value, int):
            raise TypeError("values must be integers")
    return values


def _in_both(value: int, right: List[int]) -> bool:
    return value in right


def sorted_intersection(left: List[int], right: List[int]) -> List[int]:
    """Return sorted unique integers present in both input lists.

    The output should contain each shared integer once, regardless of duplicate
    appearances in either input, and should be sorted ascending.
    """
    left = _require_int_list("left", left)
    right = _require_int_list("right", right)
    common = []
    for value in left:
        if _in_both(value, right):
            common.append(value)
    return sorted(common)
