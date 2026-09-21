"""Peer-verdict and own-answer submission tools."""

import json
from typing import Any

from experiments.protocol.errors import error_string, listed_positions
from experiments.protocol.state import (
    VALID_CODE_ANSWERS,
    VALID_VERDICTS,
    _log_event,
    agent_state,
    peer_id,
)


def _answer_format_error(
    state: dict[str, Any],
    slot: dict[str, Any],
    answer: str,
) -> str | None:
    """Validate the answer against the submitting agent's task type and answer key."""
    task_type = str(state.get("task_type", "code_analysis"))
    if task_type == "code_analysis":
        if answer not in VALID_CODE_ANSWERS:
            return "code answer must be no_bug or bug"
        return None

    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError as exc:
        return f"answer must contain valid JSON: {exc}"
    if not isinstance(parsed, list):
        return "answer JSON must be an array"

    if task_type == "record_extraction":
        malformed = [
            index
            for index, item in enumerate(parsed)
            if not isinstance(item, str) or not item.strip()
        ]
        if malformed:
            return (
                "extraction answer must be an array of non-empty record-ID strings; "
                f"invalid indexes: {listed_positions(malformed)}"
            )
        return None

    if task_type == "data_search":
        answer_key = [str(column) for column in slot.get("answer_key", [])]
        if not answer_key:
            return "data-search task is missing answer_key"
        answer_key_set = set(answer_key)
        malformed: list[str] = []
        for index, row in enumerate(parsed):
            if not isinstance(row, dict):
                malformed.append(f"row[{index}] is not an object")
                continue
            row_columns = set(row)
            missing = sorted(answer_key_set - row_columns)
            unexpected = sorted(row_columns - answer_key_set)
            if missing:
                malformed.append(f"row[{index}] missing {missing}")
            if unexpected:
                malformed.append(f"row[{index}] has unexpected {unexpected}")
        if malformed:
            return (
                "data-search answer rows must contain exactly the answer_key columns; "
                + listed_positions(malformed, separator="; ")
            )
        return None

    # The dispatch gate rejects unknown task types before reaching this handler.
    raise ValueError(f"Unknown task_type: {task_type}")


def _submit_verdict(
    state: dict[str, Any], actor: str, verdict: str, reasoning: str
) -> dict[str, Any]:
    """Record a verdict on the peer's task.

    The runner waits for both verdicts or exhausted attempt budgets before scoring.
    """
    slot = agent_state(state, actor)
    target = peer_id(actor)

    if verdict not in VALID_VERDICTS:
        result = {
            "recorded": False,
            "error": error_string("VerdictError", f"not a verdict: {verdict}"),
        }
        _log_event(
            state,
            {
                "actor": actor,
                "target": target,
                "tool": "submit_verdict",
                "success": False,
                "error": result["error"],
            },
        )
        return result

    if slot["verdict_submission"] is not None:
        result = {
            "recorded": False,
            "error": error_string(
                "VerdictError",
                f"{actor} already submitted its verdict for this episode",
            ),
        }
        _log_event(
            state,
            {
                "actor": actor,
                "target": target,
                "tool": "submit_verdict",
                "success": False,
                "error": result["error"],
            },
        )
        return result

    slot["verdict_submission"] = {
        "verdict": verdict,
        "reasoning": reasoning,
        "target": target,
    }
    _log_event(
        state,
        {
            "actor": actor,
            "target": target,
            "tool": "submit_verdict",
            "success": True,
            "verdict": verdict,
        },
    )
    return {"recorded": True, "verdict": verdict, "target": target}


def _save_answer(
    state: dict[str, Any],
    actor: str,
    slot: dict[str, Any],
    *,
    artifact_key: str,
    answer: str,
    reasoning: str,
) -> dict[str, Any]:
    if not reasoning.strip():
        result = {
            "success": False,
            "error": error_string(
                "InvalidAnswerFormatError", "reasoning must be non-empty"
            ),
        }
        _log_event(
            state,
            {
                "actor": actor,
                "tool": "save_final_answer",
                "success": False,
                "error": result["error"],
            },
        )
        return result

    format_error = _answer_format_error(state, slot, answer)
    if format_error is not None:
        result = {
            "success": False,
            "error": error_string("InvalidAnswerFormatError", format_error),
        }
        _log_event(
            state,
            {
                "actor": actor,
                "tool": "save_final_answer",
                "success": False,
                "error": result["error"],
            },
        )
        return result

    slot[artifact_key] = {"reasoning": reasoning, "answer": answer}
    # Save the complete answer without channel-specific length metadata.
    result = {"success": True}
    _log_event(
        state,
        {
            "actor": actor,
            "tool": "save_final_answer",
            "success": True,
        },
    )
    return result


def _save_final_answer(
    state: dict[str, Any],
    actor: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    slot = agent_state(state, actor)
    task_type = str(state.get("task_type", "code_analysis"))
    answer = str(arguments.get("answer", ""))
    reasoning = str(arguments.get("reasoning", ""))
    artifact_keys = {
        "record_extraction": "extraction_artifact",
        "data_search": "data_search_artifact",
        "code_analysis": "code_assessment",
    }
    if task_type not in artifact_keys:
        # Unknown task types are rejected by dispatch.
        raise ValueError(f"Unknown task_type: {task_type}")
    result = _save_answer(
        state=state,
        actor=actor,
        slot=slot,
        artifact_key=artifact_keys[task_type],
        answer=answer,
        reasoning=reasoning,
    )
    if result.get("success") is True:
        slot["final_answer_saved"] = True
    return result
