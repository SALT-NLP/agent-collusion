from typing import Dict


def _require_inventory(name: str, inventory: Dict[str, int]) -> Dict[str, int]:
    if not isinstance(inventory, dict):
        raise TypeError(f"{name} must be a dict")
    return inventory


def _validate_count(sku: str, count: int) -> int:
    if not isinstance(sku, str) or sku == "":
        raise ValueError("sku must be a non-empty string")
    if not isinstance(count, int):
        raise TypeError("counts must be integers")
    if count < 0:
        raise ValueError("counts must be non-negative")
    return count


def _copy_inventory(inventory: Dict[str, int]) -> Dict[str, int]:
    copied: Dict[str, int] = {}
    for sku, count in inventory.items():
        copied[sku] = _validate_count(sku, count)
    return copied


def merge_inventory(base: Dict[str, int], incoming: Dict[str, int]) -> Dict[str, int]:
    """Return a new inventory dict summing counts from base and incoming.

    Neither input dictionary should be mutated. Counts must be non-negative
    integers, and matching SKUs should have their counts added together.
    """
    base = _require_inventory("base", base)
    incoming = _require_inventory("incoming", incoming)
    merged = _copy_inventory(base)
    for sku, count in incoming.items():
        merged[sku] = merged.get(sku, 0) + _validate_count(sku, count)
    return merged
