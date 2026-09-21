"""Expand per-episode verdict-policy and throttling schedules."""

from collections.abc import Callable
from dataclasses import dataclass

VERDICT_POLICIES = ("raw-only", "summary-allowed")
THROTTLE_POLICIES = ("no-throttle", "throttled")


@dataclass(frozen=True)
class PolicySpan:
    policy: str
    count: int | None


def _validate_verdict_policy(verdict_policy: str) -> str:
    if verdict_policy not in VERDICT_POLICIES:
        raise ValueError(f"Unknown verdict_policy: {verdict_policy}")
    return verdict_policy


def _expand_policy_spans(
    spans: str,
    episode_count: int,
    default_policy: str,
    *,
    validate: Callable[[str], str],
    option_name: str,
    example: str,
    notice: Callable[[str], None] | None = None,
) -> list[str]:
    default_policy = validate(default_policy)
    if not spans.strip():
        return [default_policy] * episode_count

    expanded: list[str] = []
    saw_rest = False
    for raw_part in spans.split(","):
        part = raw_part.strip()
        if not part:
            continue
        span = _parse_policy_span(
            part,
            validate=validate,
            option_name=option_name,
            example=example,
        )
        if saw_rest:
            raise ValueError(f"The '*' span in {option_name} must be the final span.")
        if span.count is None:
            expanded.extend(
                [span.policy] * max(episode_count - len(expanded), 0),
            )
            saw_rest = True
            continue
        expanded.extend([span.policy] * span.count)

    # Reject schedules that leave episodes without an explicit policy.
    if len(expanded) < episode_count:
        raise ValueError(
            f"{option_name} covers {len(expanded)} episode(s), but this run selected "
            f"{episode_count}. End with a '*' span to take the rest, as in {example}."
        )
    # Trim excess spans to the episode count after task limiting and pairing.
    if len(expanded) > episode_count and notice is not None:
        notice(
            f"{option_name} covers {len(expanded)} episode(s) and this run selected "
            f"{episode_count}; the first {episode_count} are used."
        )
    return expanded[:episode_count]


def _parse_policy_span(
    text: str,
    *,
    validate: Callable[[str], str],
    option_name: str,
    example: str,
) -> PolicySpan:
    try:
        policy, count_text = [piece.strip() for piece in text.split(":", 1)]
    except ValueError as exc:
        raise ValueError(
            f"Invalid {option_name} format. Use entries like {example}."
        ) from exc
    policy = validate(policy)
    if count_text == "*":
        return PolicySpan(policy=policy, count=None)
    try:
        count = int(count_text)
    except ValueError as exc:
        raise ValueError(f"Invalid span count in {option_name}: {count_text}") from exc
    if count < 0:
        raise ValueError(f"Span count must be non-negative: {count_text}")
    return PolicySpan(policy=policy, count=count)


def expand_verdict_policy_spans(
    verdict_policy_spans: str,
    episode_count: int,
    default_policy: str,
    notice: Callable[[str], None] | None = None,
) -> list[str]:
    return _expand_policy_spans(
        verdict_policy_spans,
        episode_count,
        default_policy,
        validate=_validate_verdict_policy,
        option_name="--verdict-policy-spans",
        example="raw-only:2,summary-allowed:3,raw-only:*",
        notice=notice,
    )


def _validate_throttle_policy(policy: str) -> str:
    """Validate the canonical throttling-policy name."""
    if policy not in THROTTLE_POLICIES:
        raise ValueError(
            f"Unknown throttle policy: {policy!r}; expected one of "
            + ", ".join(THROTTLE_POLICIES)
        )
    return policy


def expand_throttle_spans(
    throttle_spans: str,
    episode_count: int,
    default_policy: str = "throttled",
    notice: Callable[[str], None] | None = None,
) -> list[str]:
    return _expand_policy_spans(
        throttle_spans,
        episode_count,
        default_policy,
        validate=_validate_throttle_policy,
        option_name="--throttle-spans",
        example="no-throttle:2,throttled:*",
        notice=notice,
    )
