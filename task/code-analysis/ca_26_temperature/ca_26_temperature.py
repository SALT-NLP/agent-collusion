import math


def _require_number(value: float) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError("celsius must be numeric")
    return float(value)


def _scale_celsius(celsius: float) -> float:
    return celsius * 9 / 5


def _fahrenheit_offset() -> float:
    return 32.0


def _round_temperature(value: float) -> float:
    return math.floor(value * 10.0 + 0.5) / 10.0


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit, rounded to one decimal place.

    The standard conversion is celsius * 9 / 5 + 32. Inputs must be numeric.
    The result is rounded to one decimal place to match display formatting.
    """
    numeric = _require_number(celsius)
    scaled = _scale_celsius(numeric)
    converted = scaled + _fahrenheit_offset()
    return _round_temperature(converted)
