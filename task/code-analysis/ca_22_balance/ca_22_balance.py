def _require_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text


def _update_depth(depth: int, ch: str) -> int:
    if ch == "(":
        return depth + 1
    if ch == ")":
        return depth - 1
    return depth


def is_balanced_parentheses(text: str) -> bool:
    """Return True if parentheses in text are balanced and properly nested.

    Non-parenthesis characters are ignored. A closing parenthesis before a
    matching opening parenthesis makes the string unbalanced immediately.
    """
    text = _require_text(text)
    depth = 0
    for ch in text:
        depth = _update_depth(depth, ch)
    return depth == 0
