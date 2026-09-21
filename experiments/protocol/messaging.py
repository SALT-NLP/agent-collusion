"""Inter-agent messaging and per-agent raw-log retrieval."""

import json
from typing import Any

from experiments.protocol.errors import error_string
from experiments.protocol.state import _log_event, agent_state, peer_id


def _send_message(
    state: dict[str, Any],
    actor: str,
    content: str,
    message_type: str = "other",
) -> dict[str, Any]:
    slot = agent_state(state, actor)
    if slot["delivered"]:
        return _reject_message(
            state=state,
            actor=actor,
            content=content,
            message_type=message_type,
            error=error_string(
                "MessageError",
                f"{actor} already sent its one message for this round",
            ),
        )

    if (
        state["throttled"]
        and len(content) > state["char_limit"]
    ):
        return _reject_message(
            state=state,
            actor=actor,
            content=content,
            message_type=message_type,
            error=error_string(
                "MessageError", f"exceeds {state['char_limit']} char limit"
            ),
        )

    slot["delivered"] = True
    recipient = peer_id(actor)
    # Keep the originating 0-based round; display tags add one.
    payload: dict[str, Any] = {
        "from": actor,
        "to": recipient,
        "content": content,
        "round": state["round"],
    }
    payload["message_type"] = message_type
    # Deliver immediately so the peer can reply within the same round.
    agent_state(state, recipient)["inbox"].append(payload)

    result = {"success": True, "chars_sent": len(content)}
    event = {
        "actor": actor,
        "recipient": recipient,
        "tool": "send_message",
        "success": True,
        "content": content,
    }
    event["message_type"] = message_type
    _log_event(state, event)
    return result


def _reject_message(
    *,
    state: dict[str, Any],
    actor: str,
    content: str,
    message_type: str,
    error: str,
) -> dict[str, Any]:
    result = {"success": False, "error": error, "chars_sent": 0}
    # Record the complete attempted message, including refused deliveries.
    event = {
        "actor": actor,
        "recipient": peer_id(actor),
        "tool": "send_message",
        "success": False,
        "error": error,
        "content": content,
    }
    event["message_type"] = message_type
    _log_event(state, event)
    return result


def _format_private_raw_log(slot: dict[str, Any]) -> str:
    return json.dumps(
        slot.get("private_raw_log", []),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _get_log(state: dict[str, Any], actor: str) -> dict[str, Any]:
    """Return only the calling agent's raw log.

    The dispatch gate permits one retrieval per agent per episode.
    """
    slot = agent_state(state, actor)
    raw_log = _format_private_raw_log(slot)
    slot["private_raw_log_retrieved"] = True
    result = {
        "success": True,
        "raw_log": raw_log,
        "raw_log_chars": len(raw_log),
    }
    _log_event(
        state,
        {
            "actor": actor,
            "tool": "get_log",
            "success": True,
        },
    )
    return result
