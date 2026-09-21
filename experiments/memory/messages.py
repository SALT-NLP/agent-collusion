"""Episode-aligned message-memory operations shared by runner and agents."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# Internal episode and phase boundaries, removed from model requests and snapshots.
INTERNAL_ACTIVE_EPISODE_ROLE = "_runner_active_episode_start"
INTERNAL_EPISODE_BOUNDARY_ROLE = "_runner_episode_memory_boundary"
INTERNAL_OUTCOME_FEEDBACK_ROLE = "_runner_outcome_feedback_start"
INTERNAL_COMMUNICATION_START_ROLE = "_runner_communication_phase_start"


def active_episode_start_message() -> dict[str, Any]:
    """Create the runner-only delimiter for the episode now being worked on."""
    return {"role": INTERNAL_ACTIVE_EPISODE_ROLE}


def outcome_feedback_start_message() -> dict[str, Any]:
    """Create an internal outcome boundary independent of message text."""
    return {"role": INTERNAL_OUTCOME_FEEDBACK_ROLE}


def communication_phase_start_message() -> dict[str, Any]:
    """Create an internal boundary immediately before the communication brief."""
    return {"role": INTERNAL_COMMUNICATION_START_ROLE}


def episode_memory_boundary_message(episode_id: str) -> dict[str, Any]:
    """Create a runner-only delimiter for one remembered episode."""
    return {
        "role": INTERNAL_EPISODE_BOUNDARY_ROLE,
        "episode_id": str(episode_id),
    }


def is_internal_episode_boundary(message: dict[str, Any]) -> bool:
    return message.get("role") == INTERNAL_EPISODE_BOUNDARY_ROLE


def is_internal_outcome_feedback_start(message: dict[str, Any]) -> bool:
    return message.get("role") == INTERNAL_OUTCOME_FEEDBACK_ROLE


def is_internal_communication_phase_start(message: dict[str, Any]) -> bool:
    return message.get("role") == INTERNAL_COMMUNICATION_START_ROLE


def is_internal_marker(message: dict[str, Any]) -> bool:
    return message.get("role") in (
        INTERNAL_ACTIVE_EPISODE_ROLE,
        INTERNAL_EPISODE_BOUNDARY_ROLE,
        INTERNAL_OUTCOME_FEEDBACK_ROLE,
        INTERNAL_COMMUNICATION_START_ROLE,
    )


def model_visible_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return messages that may be sent to a model, preserving their order."""
    return [message for message in messages if not is_internal_marker(message)]


def snapshot_visible_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deep-copy model-visible history for persisted run output."""
    return deepcopy(model_visible_messages(messages))


def restore_episode_boundaries(
    initial_messages: list[dict[str, Any]],
    episode_snapshots: list[tuple[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Restore episode boundaries from persisted model-visible history snapshots.

    Locate earlier episode endings in the newest snapshot, allowing for oldest-block
    drops during context-limit retries. Within-episode phase markers are not restored.
    """
    if not episode_snapshots:
        return deepcopy(initial_messages)

    def is_prefix(shorter: list[dict[str, Any]], longer: list[dict[str, Any]]) -> bool:
        return len(shorter) <= len(longer) and all(
            shorter[index] == longer[index] for index in range(len(shorter))
        )

    _, newest = episode_snapshots[-1]
    if not is_prefix(initial_messages, newest):
        raise ValueError(
            "The opening turns are not a prefix of the final history, so these "
            "snapshots are not this run's"
        )

    # Locate earlier episode endings by their final turns; dropped blocks shift offsets.
    def locate_end(snapshot: list[dict[str, Any]]) -> int | None:
        if not snapshot:
            return None
        if is_prefix(snapshot, newest):
            return len(snapshot)
        # Match three turns to reduce ambiguity from repeated single messages.
        window = snapshot[-3:]
        for end in range(len(newest), len(window) - 1, -1):
            if newest[end - len(window) : end] == window:
                return end
        return None

    cuts: list[tuple[int, str]] = []
    for episode_id, snapshot in episode_snapshots[:-1]:
        end = locate_end(snapshot)
        if end is not None and end < len(newest):
            cuts.append((end, str(episode_id)))
    cuts = sorted(set(cuts))

    restored = deepcopy(initial_messages)
    consumed = len(initial_messages)
    # Open each block after the preceding surviving episode boundary.
    names = [episode_snapshots[0][0]] + [name for _, name in cuts]
    offsets = [consumed] + [offset for offset, _ in cuts]
    for name, start in zip(names, offsets):
        if start < consumed:
            continue
        restored.append(episode_memory_boundary_message(str(name)))
        end = next((o for o in offsets if o > start), len(newest))
        restored.extend(deepcopy(newest[start:end]))
        consumed = end
    if consumed < len(newest):
        restored.extend(deepcopy(newest[consumed:]))
    return restored


def active_episode_index(messages: list[dict[str, Any]]) -> int | None:
    for index, message in enumerate(messages):
        if message.get("role") == INTERNAL_ACTIVE_EPISODE_ROLE:
            return index
    return None


def outcome_feedback_start_index(messages: list[dict[str, Any]]) -> int | None:
    """Where the outcome feedback begins, as an index into one episode's turns."""
    for index, message in enumerate(messages):
        if is_internal_outcome_feedback_start(message):
            return index
    return None


def communication_phase_start_index(messages: list[dict[str, Any]]) -> int | None:
    """Where the communication phase opens, as an index into one episode's turns."""
    for index, message in enumerate(messages):
        if is_internal_communication_phase_start(message):
            return index
    return None


def episode_boundary_indices(messages: list[dict[str, Any]]) -> list[int]:
    """Where each remembered episode starts, by its runner-only delimiter."""
    return [
        index
        for index, message in enumerate(messages)
        if is_internal_episode_boundary(message)
    ]


def _memory_block_bounds(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """The (start, end) span of every remembered episode, oldest first."""
    starts = episode_boundary_indices(messages)
    if not starts:
        return []
    active = active_episode_index(messages)
    ends = starts[1:] + [active if active is not None else len(messages)]
    return list(zip(starts, ends))


def drop_oldest_episode_block(messages: list[dict[str, Any]]) -> int:
    """Drop the oldest remembered episode, leaving the active one intact."""
    blocks = _memory_block_bounds(messages)
    if not blocks:
        return 0
    start, end = blocks[0]
    del messages[start:end]
    return end - start


def limit_episode_blocks(messages: list[dict[str, Any]], max_episodes: int) -> None:
    """Keep at most the newest ``max_episodes`` remembered episodes."""
    if max_episodes < -1:
        raise ValueError("max_episodes must be -1 or greater")
    if max_episodes == -1:
        return
    blocks = _memory_block_bounds(messages)
    if len(blocks) <= max_episodes:
        return
    # Drop everything before the retained episode blocks.
    cut_to = blocks[-max_episodes][0] if max_episodes else blocks[-1][1]
    del messages[blocks[0][0] : cut_to]
