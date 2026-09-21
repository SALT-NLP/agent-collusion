from typing import List


def _require_lines(lines: List[str]) -> List[str]:
    if not isinstance(lines, list):
        raise TypeError("lines must be a list")
    for line in lines:
        if not isinstance(line, str):
            raise TypeError("lines must contain strings")
    return lines


def _clean_line(line: str) -> str:
    return line.strip()


def _should_keep(line: str) -> bool:
    return True


def compact_whitespace(lines: List[str]) -> List[str]:
    """Strip each line and omit empty results.

    Whitespace-only lines should not appear in the output. The relative order of
    non-empty stripped lines should be preserved.
    """
    lines = _require_lines(lines)
    result = []
    for line in lines:
        stripped = _clean_line(line)
        if _should_keep(stripped):
            result.append(stripped)
    return result
