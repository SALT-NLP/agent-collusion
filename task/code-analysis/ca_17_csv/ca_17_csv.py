from typing import List


def _require_line(line: str) -> str:
    if not isinstance(line, str):
        raise TypeError("line must be a string")
    return line


def _split_fields(line: str) -> List[str]:
    return line.split(",")


def _trim_field(field: str) -> str:
    return field.strip()


def parse_csv_line(line: str) -> List[str]:
    """Parse a simple comma-separated line, trimming spaces around each field.

    This parser intentionally does not implement quoted CSV. Empty fields are
    preserved, and surrounding whitespace around every field is stripped.
    """
    line = _require_line(line)
    fields = []
    for field in _split_fields(line):
        fields.append(_trim_field(field))
    return fields
