from typing import Tuple


def _require_email(email: str) -> str:
    if not isinstance(email, str):
        raise TypeError("email must be a string")
    return email.strip()


def _split_email(email: str) -> Tuple[str, str]:
    if email.count("@") != 1:
        raise ValueError("email must contain one @")
    local, domain = email.split("@")
    if not local or not domain:
        raise ValueError("email must include local and domain parts")
    return local, domain


def _mask_local(local: str) -> str:
    if len(local) == 1:
        return local[0] + "***"
    return local[0] + "***"


def mask_email(email: str) -> str:
    """Mask an email as first-letter + *** + @domain.

    The domain should remain unchanged. For example, user@example.com becomes
    u***@example.com. Invalid addresses with missing local or domain parts raise
    ValueError.
    """
    email = _require_email(email)
    local, domain = _split_email(email)
    masked_local = _mask_local(local)
    return masked_local + domain
