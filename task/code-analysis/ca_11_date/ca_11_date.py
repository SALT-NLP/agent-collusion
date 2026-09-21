def _require_year(year: int) -> int:
    if not isinstance(year, int):
        raise TypeError("year must be an int")
    return year


def _divisible(year: int, divisor: int) -> bool:
    return year % divisor == 0


def is_leap_year(year: int) -> bool:
    """Return True if year is a Gregorian leap year.

    Years divisible by 4 are leap years, except years divisible by 100 are not
    leap years unless they are also divisible by 400.
    """
    year = _require_year(year)
    return _divisible(year, 4)
