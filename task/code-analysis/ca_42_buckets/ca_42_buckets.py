from typing import Dict, List


def _require_words(words: List[str]) -> List[str]:
    if not isinstance(words, list):
        raise TypeError("words must be a list")
    for word in words:
        if not isinstance(word, str):
            raise TypeError("words must contain strings")
    return words


def _bucket_key(word: str) -> int:
    return len(word)


def _place_word(buckets: Dict[int, List[str]], word: str) -> None:
    key = _bucket_key(word)
    buckets[key] = [word]


def bucket_by_length(words: List[str]) -> Dict[int, List[str]]:
    """Group words by their length while preserving input order within each bucket.

    Empty strings are allowed and belong to bucket 0. The returned dictionary
    associates each length with all words of that length in their original order.
    """
    words = _require_words(words)
    buckets: Dict[int, List[str]] = {}
    for word in words:
        _place_word(buckets, word)
    return buckets
