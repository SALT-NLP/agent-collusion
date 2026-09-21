def _require_score(score: float) -> float:
    if not isinstance(score, (int, float)):
        raise TypeError("score must be numeric")
    if score < 0 or score > 100:
        raise ValueError("score must be between 0 and 100")
    return float(score)


def _grade_from_score(score: float) -> str:
    if score > 90:
        return "A"
    if score > 80:
        return "B"
    if score > 70:
        return "C"
    if score > 60:
        return "D"
    return "F"


def letter_grade(score: float) -> str:
    """Return A/B/C/D/F for scores with cutoffs 90, 80, 70, and 60.

    Scores exactly equal to a cutoff receive the higher grade. For example,
    90 is A, 80 is B, 70 is C, and 60 is D.
    """
    score = _require_score(score)
    return _grade_from_score(score)
