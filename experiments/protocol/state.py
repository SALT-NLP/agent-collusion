"""Shared episode state, per-agent task state, and event logging.

Phase, round, channel limits, and events are shared. Task resources, progress,
and verdicts live in ``state["agents"][agent_id]``.
"""

from typing import Any

# Agent order also determines who opens each communication round.
AGENT_IDS = ("alice", "bob")

# Use display names in prose and agent IDs in tags and routing.
AGENT_DISPLAY_NAMES = {agent_id: agent_id.capitalize() for agent_id in AGENT_IDS}

# Shared phase barriers; agents alternate only within communication.
PHASES = ("task", "communication", "verdict")

# Peer verdicts use accept/reject, distinct from task evaluation pass/fail.
VALID_VERDICTS = {"accept", "reject"}
# Code assessments describe the target, not whether the agent did its task correctly.
VALID_CODE_ANSWERS = {"no_bug", "bug"}

# Translate manifest ground truth into agent-facing wording.
VERDICT_BY_EXPECTED_VERDICT = {"pass": "accept", "fail": "reject"}

# Derive the successful-task verdict from the shared mapping.
ACCEPT_VERDICT = VERDICT_BY_EXPECTED_VERDICT["pass"]

# Reward per qualifying verdict; the reward scheme determines how many terms are paid.
VERDICT_REWARD = 10


def peer_id(agent_id: str) -> str:
    """The other agent in the pair -- the one this agent judges and is judged by."""
    if agent_id == AGENT_IDS[0]:
        return AGENT_IDS[1]
    if agent_id == AGENT_IDS[1]:
        return AGENT_IDS[0]
    raise ValueError(f"Unknown agent: {agent_id}")


def display_name(agent_id: str) -> str:
    """One agent's name as every prompt spells it. Raises for an unknown id."""
    try:
        return AGENT_DISPLAY_NAMES[agent_id]
    except KeyError:
        raise ValueError(f"Unknown agent: {agent_id}") from None


def agent_state(state: dict[str, Any], agent_id: str) -> dict[str, Any]:
    """Return an existing agent slot; raise for unknown agent IDs."""
    try:
        return state["agents"][agent_id]
    except KeyError:
        raise ValueError(f"Unknown agent: {agent_id}") from None


def verdict_is_correct(final_verdict: str, expected_verdict: str) -> bool:
    """Compare a peer verdict with ground truth through the verdict vocabulary mapping."""
    return final_verdict == VERDICT_BY_EXPECTED_VERDICT.get(expected_verdict)


def _agent_slot() -> dict[str, Any]:
    """One agent's own half of an episode: its task, its work, its verdict."""
    return {
        # Agent-visible display ID; the internal task ID remains in the manifest task.
        "display_id": "",
        "target_code_path": "",
        "function_name": "",
        "source_path": "",
        "database_path": "",
        "answer_key": [],
        "task_tmp_dir": "",
        "written_test_files": [],
        # what this agent produced
        "extraction_artifact": None,
        "data_search_artifact": None,
        "code_assessment": None,
        "private_raw_log": [],
        "final_answer_saved": False,
        "private_raw_log_retrieved": False,
        # this agent's verdict on its peer's task
        "verdict_submission": None,
        # Unread messages, each carrying its originating communication round.
        "inbox": [],
        "delivered": False,
    }


def create_channel_state(
    episode_id: str,
    throttled: bool = True,
    char_limit: int = 200,
) -> dict[str, Any]:
    """Create shared episode state with an episode-wide communication constraint."""
    return {
        "episode_id": episode_id,
        # Internal rounds are 0-based; displayed rounds are 1-based.
        "round": 0,
        "phase": PHASES[0],
        "throttled": throttled,
        "char_limit": char_limit,
        "repo_root": "",
        # Agents share a task type but have separate task instances.
        "task_type": "code_analysis",
        "events": [],
        "agents": {agent_id: _agent_slot() for agent_id in AGENT_IDS},
    }


def set_phase(state: dict[str, Any], phase: str) -> None:
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    state["phase"] = phase


def deliver_runner_notice(
    state: dict[str, Any],
    recipient: str,
    content: str,
) -> None:
    """Queue a runner notice with its originating round for the agent's next turn."""
    agent_state(state, recipient)["inbox"].append(
        {
            "from": "runner",
            "to": recipient,
            "message_type": "other",
            "content": content,
            "round": state["round"],
        }
    )


def pop_incoming_messages(state: dict[str, Any], actor: str) -> list[dict[str, Any]]:
    slot = agent_state(state, actor)
    messages = slot["inbox"]
    slot["inbox"] = []
    return messages


def _log_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    event["event_id"] = len(state["events"])
    event["round"] = state["round"]
    event.setdefault("phase", state["phase"])
    state["events"].append(event)


def reset_message_delivery(state: dict[str, Any], actor: str) -> None:
    """Start a new delivery attempt for one agent."""
    agent_state(state, actor)["delivered"] = False
