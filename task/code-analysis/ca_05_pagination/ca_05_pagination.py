from typing import Any, List


def _require_items(items: List[Any]) -> List[Any]:
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    return items


def _require_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError("page and per_page must be positive")
    return value


def _page_bounds(page: int, per_page: int) -> tuple[int, int]:
    start = (page - 1) * per_page
    end = start + per_page
    return start, end


def paginate(items: List[Any], page: int, per_page: int) -> List[Any]:
    """Return the 1-indexed page slice for items.

    Page 1 starts at the first item. Pages beyond the end return an empty list.
    page and per_page must both be positive.
    """
    items = _require_items(items)
    page = _require_positive_int("page", page)
    per_page = _require_positive_int("per_page", per_page)
    start, end = _page_bounds(page, per_page)
    return items[start:end]
