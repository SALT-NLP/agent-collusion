import re
from typing import List


def _require_phrase(phrase: str) -> str:
    if not isinstance(phrase, str):
        raise TypeError("phrase must be a string")
    return phrase


def _word_candidates(phrase: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", phrase)


def _is_word(candidate: str) -> bool:
    return candidate.isalpha()


def _initial(candidate: str) -> str:
    return candidate[0].upper()


def make_acronym(phrase: str) -> str:
    """Return an uppercase acronym from alphabetic words in phrase.

    Punctuation and numeric fragments should not prevent extracting alphabetic
    words. For example, "central-processing unit" should produce CPU.
    """
    phrase = _require_phrase(phrase)
    letters = []
    for candidate in _word_candidates(phrase):
        if _is_word(candidate):
            letters.append(_initial(candidate))
    return "".join(letters)
