def _require_digits(digits: str) -> str:
    if not isinstance(digits, str):
        raise TypeError("digits must be a string")
    if len(digits) != 10 or not digits.isdigit():
        raise ValueError("digits must contain exactly 10 digits")
    return digits


def _area_code(digits: str) -> str:
    return digits[:3]


def _prefix(digits: str) -> str:
    return digits[3:6]


def _line_number(digits: str) -> str:
    return digits[6:]


def format_phone(digits: str) -> str:
    """Format exactly 10 digits as (XXX) XXX-XXXX.

    Only plain digit strings are accepted. Separators, spaces, country codes,
    and missing digits should raise ValueError rather than being normalized.
    """
    digits = _require_digits(digits)
    area = _area_code(digits)
    prefix = _prefix(digits)
    line = _line_number(digits)
    return f"({area}) {prefix}-{line}"
