import math
from typing import Iterable, List


def _validate_tax_rate(tax_rate: float) -> float:
    if not isinstance(tax_rate, (int, float)):
        raise TypeError("tax_rate must be numeric")
    if tax_rate < 0:
        raise ValueError("tax_rate must be non-negative")
    return float(tax_rate)


def _collect_prices(prices: Iterable[float]) -> List[float]:
    collected = []
    for price in prices:
        if not isinstance(price, (int, float)):
            raise TypeError("prices must be numeric")
        if price < 0:
            raise ValueError("prices must be non-negative")
        collected.append(float(price))
    return collected


def _subtotal(prices: List[float]) -> float:
    total = 0.0
    for price in prices:
        total += price
    return total


def calculate_cart_total(prices: Iterable[float], tax_rate: float) -> float:
    """Return the sum of prices plus tax, rounded to two decimals.

    tax_rate is expressed as a decimal, so 0.0825 means 8.25 percent tax.
    Empty carts return 0.0. Negative prices and negative tax rates are invalid.
    """
    rate = _validate_tax_rate(tax_rate)
    collected = _collect_prices(prices)
    subtotal = _subtotal(collected)
    return math.floor(subtotal * (1 + rate) * 100.0 + 0.5) / 100.0
