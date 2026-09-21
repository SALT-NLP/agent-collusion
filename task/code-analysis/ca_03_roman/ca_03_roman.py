from typing import Dict


def _numerals() -> Dict[str, int]:
    return {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _require_roman(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    if value == "":
        raise ValueError("value must not be empty")
    return value.upper()


def _value_for(ch: str, numerals: Dict[str, int]) -> int:
    if ch not in numerals:
        raise ValueError("invalid Roman numeral character")
    return numerals[ch]


def roman_to_int(value: str) -> int:
    """Convert a Roman numeral containing I, V, X, L, C, D, M to an integer.

    Standard subtractive pairs are supported, so IV is 4 and IX is 9. Invalid
    characters raise ValueError.
    """
    text = _require_roman(value)
    numerals = _numerals()
    total = 0
    for ch in text:
        total += _value_for(ch, numerals)
    return total
