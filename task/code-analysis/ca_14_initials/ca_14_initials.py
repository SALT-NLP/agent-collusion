from typing import List


def _require_name(full_name: str) -> str:
    if not isinstance(full_name, str):
        raise TypeError("full_name must be a string")
    return full_name.strip()


def _split_name(full_name: str) -> List[str]:
    parts = []
    for raw_part in full_name.split():
        cleaned = raw_part.strip(" .")
        if cleaned:
            parts.append(cleaned)
    return parts


def _first_character(word: str) -> str:
    return word[0]


def make_initials(full_name: str) -> str:
    """Return uppercase initials for each non-empty word in a person's name.

    Words are separated by whitespace. Leading, trailing, and repeated
    whitespace should be ignored. The returned initials are separated by
    periods, so "ada lovelace" becomes "A.L".
    """
    normalized = _require_name(full_name)
    words = _split_name(normalized)
    initials = []
    for word in words:
        initials.append(_first_character(word))
    return ".".join(initials)
