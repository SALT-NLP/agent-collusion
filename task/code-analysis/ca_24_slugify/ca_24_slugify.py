import re
from typing import List


def _require_title(title: str) -> str:
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    return title.strip()


def _remove_punctuation(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9\s-]", "", title)


def _to_words(title: str) -> List[str]:
    words = []
    for part in title.lower().split(" "):
        if part:
            words.append(part)
    return words


def _join_words(words: List[str]) -> str:
    return "-".join(words)


def slugify_title(title: str) -> str:
    """Return a lowercase URL slug with words separated by single hyphens.

    Punctuation is removed, repeated whitespace or hyphens collapse to one
    hyphen, and leading or trailing separators are omitted from the result.
    """
    required = _require_title(title)
    cleaned = _remove_punctuation(required)
    words = _to_words(cleaned)
    return _join_words(words)
