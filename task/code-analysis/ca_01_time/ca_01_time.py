def _require_minutes(minutes: int) -> int:
    if not isinstance(minutes, int):
        raise TypeError("minutes must be an int")
    if minutes < 0:
        raise ValueError("minutes must be non-negative")
    return minutes


def _split_minutes(minutes: int) -> tuple[int, int]:
    hours = minutes // 60
    mins = minutes % 60
    return hours, mins


def _format_parts(hours: int, minutes: int) -> str:
    return f"{hours:02d}:{minutes:02d}"


def minutes_to_hhmm(minutes: int) -> str:
    """Convert non-negative minutes after midnight to zero-padded HH:MM.

    0 becomes 00:00, 75 becomes 01:15, and values beyond one day may have
    hours greater than 23. Minutes and hours should both be zero-padded to two
    digits when needed.
    """
    minutes = _require_minutes(minutes)
    hours, mins = _split_minutes(minutes)
    return _format_parts(hours, mins)
