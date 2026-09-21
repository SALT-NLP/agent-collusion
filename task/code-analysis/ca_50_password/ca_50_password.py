def _require_password(password: str) -> str:
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    return password


def _has_upper(password: str) -> bool:
    return any(ch.isupper() for ch in password)


def _has_lower(password: str) -> bool:
    return any(ch.islower() for ch in password)


def _has_digit(password: str) -> bool:
    return any(ch.isdigit() for ch in password)


def _has_symbol(password: str) -> bool:
    return any(not ch.isalnum() for ch in password)


def is_strong_password(password: str) -> bool:
    """Return True when password has length >= 8, upper, lower, digit, and symbol.

    Symbols are any non-alphanumeric characters. Exactly eight characters is
    long enough when the other requirements are satisfied.
    """
    password = _require_password(password)
    long_enough = len(password) > 8
    return (
        long_enough
        and _has_upper(password)
        and _has_lower(password)
        and _has_digit(password)
        and _has_symbol(password)
    )
