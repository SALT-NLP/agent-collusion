"""Construction and retention of each agent's memory between episodes."""

from typing import Any

from experiments.memory.messages import (
    active_episode_index,
    communication_phase_start_index,
    episode_memory_boundary_message,
    limit_episode_blocks,
    outcome_feedback_start_index,
    outcome_feedback_start_message,
)
from experiments.protocol.rewards import RewardScheme
from experiments.prompts.messages import agent_outcome_feedback_message

CROSS_EPISODE_MEMORY_SCOPES = (
    "feedback-and-reflection",
    "communication-onward",
    "full-history",
)


def append_reflection(
    messages: list[dict[str, Any]],
    prompt: str,
    reflection: str,
    replayed_reasoning: dict[str, Any] | None = None,
) -> None:
    """Persist the reflection exchange with its original roles and replayable reasoning."""
    messages.append({"role": "user", "content": prompt})
    messages.append(
        {"role": "assistant", "content": reflection, **(replayed_reasoning or {})}
    )


def append_outcome_feedback_boundary(messages: list[dict[str, Any]]) -> None:
    """Add the internal outcome boundary when no feedback message is sent."""
    messages.append(outcome_feedback_start_message())


def append_agent_outcome_feedback(
    messages: list[dict[str, Any]],
    *,
    episode_number: int,
    own_task_id: str,
    peer_task_id: str,
    peer: str,
    own_verdict: str,
    peer_expected_verdict: str,
    peer_verdict: str,
    own_expected_verdict: str,
    reward: bool = True,
    verdict_evaluation: str = "none",
    reward_scheme: RewardScheme = RewardScheme(),
) -> str:
    """Append one agent's outcome feedback using episode and task display IDs."""
    content = agent_outcome_feedback_message(
        episode_number=episode_number,
        own_task_id=own_task_id,
        peer_task_id=peer_task_id,
        peer=peer,
        own_verdict=own_verdict,
        peer_expected_verdict=peer_expected_verdict,
        peer_verdict=peer_verdict,
        own_expected_verdict=own_expected_verdict,
        reward=reward,
        verdict_evaluation=verdict_evaluation,
        reward_scheme=reward_scheme,
    )
    append_outcome_feedback_boundary(messages)
    messages.append({"role": "user", "content": content})
    return content


def _first_remembered_index(
    block: list[dict[str, Any]],
    scope: str,
    outcome: int,
) -> int:
    """Find the phase boundary where the retained episode tail begins."""
    if scope != "communication-onward":
        # Keep outcome feedback and reflection.
        return outcome
    # Keep the communication brief and all subsequent episode messages.
    start = communication_phase_start_index(block[:outcome])
    if start is None:
        # If communication never opened, fall back to outcome and reflection.
        return outcome
    return start


def _retire_active_episode(
    messages: list[dict[str, Any]], episode_id: str
) -> int | None:
    """Convert the active episode marker into a remembered-episode marker."""
    active = active_episode_index(messages)
    if active is None:
        return None
    messages[active] = episode_memory_boundary_message(episode_id)
    return active


def _trim_completed_episode(
    messages: list[dict[str, Any]],
    scope: str,
    episode_id: str,
) -> None:
    """Retain the requested episode tail without rewriting or reordering messages."""
    boundary = _retire_active_episode(messages, episode_id)
    if boundary is None or scope == "full-history":
        return
    block_start = boundary + 1
    block = messages[block_start:]
    outcome = outcome_feedback_start_index(block)
    if outcome is None:
        # A completed episode must have an outcome boundary before it can be trimmed.
        raise ValueError(
            f"Episode {episode_id or '(unknown)'} has no outcome-feedback delimiter, "
            f"so the {scope} memory cut point cannot be found"
        )
    cut = _first_remembered_index(block, scope, outcome)
    if cut:
        del messages[block_start : block_start + cut]


def set_cross_episode_memory_from_results(
    agent_messages: dict[str, list[dict[str, Any]]],
    *,
    scope: str,
    episode_id: str = "",
    max_memory_episodes_by_agent: dict[str, int] | None = None,
) -> None:
    """Apply memory retention independently to each agent's history."""
    if scope not in CROSS_EPISODE_MEMORY_SCOPES:
        raise ValueError(f"Unknown cross-episode memory scope: {scope}")
    limits = max_memory_episodes_by_agent or {}
    for agent_id, messages in agent_messages.items():
        max_memory_episodes = limits.get(agent_id, -1)
        _trim_completed_episode(messages, scope, episode_id)
        if max_memory_episodes >= 0:
            limit_episode_blocks(messages, max_memory_episodes)
