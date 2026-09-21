from typing import List, Set


def _require_items(items: List[str]) -> List[str]:
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    for item in items:
        if not isinstance(item, str):
            raise TypeError("items must contain strings")
    return items


def _seen_key(item: str) -> str:
    return item


def _add_if_new(item: str, seen: Set[str], result: List[str]) -> None:
    key = _seen_key(item)
    if key not in seen:
        seen.add(key)
        result.append(item)


def unique_preserve_order(items: List[str]) -> List[str]:
    """Return unique strings in first-seen order, comparing case-insensitively.

    The original casing of the first occurrence should be preserved. Later
    entries that differ only by case should be treated as duplicates.
    """
    items = _require_items(items)
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        _add_if_new(item, seen, result)
    return result
