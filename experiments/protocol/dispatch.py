"""Dispatch OpenAI-compatible tool calls to protocol handlers."""

import json
from typing import Any, Callable

from experiments.protocol.code_tests import _run_tests
from experiments.protocol.databases import _inspect_database, _query_database
from experiments.protocol.errors import error_string
from experiments.protocol.files import _read_code, _read_source, _write_test_file
from experiments.protocol.messaging import _get_log, _send_message
from experiments.protocol.records import _resolve_records
from experiments.protocol.state import _log_event, agent_state
from experiments.protocol.submissions import _save_final_answer, _submit_verdict
from experiments.tool_schemas import (
    FINAL_VERDICT_TOOL_NAME,
    available_tool_names,
    task_tool_names,
)

# Exclude communication, verdict submission, and log retrieval from task raw logs.
_UNLOGGED_TOOLS = frozenset({"send_message", "get_log", FINAL_VERDICT_TOOL_NAME})


def _unavailable_tool_error(state: dict[str, Any], tool_name: str) -> str | None:
    """Check tool availability for the task type across all phases.

    Phase-specific restrictions are checked separately.
    """
    task_type = str(state.get("task_type", "code_analysis"))
    if tool_name in task_tool_names(task_type):
        return None
    return error_string(
        "ToolUnavailableError",
        f"{tool_name} is not one of the tools you are given for this task",
    )


def _task_phase_error(slot: dict[str, Any], tool_name: str) -> str | None:
    """Enforce the task sequence per agent: work, save the answer, then retrieve the log."""
    if slot["private_raw_log_retrieved"]:
        return error_string(
            "ProtocolError",
            "after get_log succeeds, no further task tool may be called this episode",
        )
    if slot["final_answer_saved"]:
        if tool_name != "get_log":
            return error_string(
                "ProtocolError",
                "after save_final_answer succeeds, only get_log may be called",
            )
        return None
    if tool_name == "get_log":
        return error_string(
            "ProtocolError",
            "save_final_answer must succeed before get_log may be called",
        )
    return None


def _phase_error(
    state: dict[str, Any],
    slot: dict[str, Any],
    tool_name: str,
) -> str | None:
    """Enforce the shared phase barrier for tool calls.

    Both agents finish task work before communication and finish communication
    before either submits a verdict.
    """
    phase = str(state["phase"])
    task_type = str(state.get("task_type", "code_analysis"))
    allowed = available_tool_names(task_type, phase)
    if tool_name not in allowed:
        listed = (
            allowed[0]
            if len(allowed) == 1
            else f"{', '.join(allowed[:-1])} and {allowed[-1]}"
        )
        return error_string(
            "PhaseError",
            f"{tool_name} is not available during the {phase} phase, "
            f"only {listed} may be called during the {phase} phase.",
        )
    if phase == "task":
        return _task_phase_error(slot, tool_name)
    return None


def _protocol_rejection(
    state: dict[str, Any],
    actor: str,
    tool_name: str,
    error: str,
) -> dict[str, Any]:
    result = {"success": False, "error": error}
    _log_event(
        state,
        {
            "actor": actor,
            "tool": tool_name,
            "success": False,
            "error": error,
        },
    )
    return result


def _dispatch_gate_error(
    state: dict[str, Any], actor: str, tool_name: str
) -> str | None:
    """Return the first tool-availability or phase error for this call."""
    unavailable = _unavailable_tool_error(state, tool_name)
    if unavailable is not None:
        return unavailable
    return _phase_error(state, agent_state(state, actor), tool_name)


def reject_undecodable_arguments(
    state: dict[str, Any],
    actor: str,
    tool_name: str,
) -> dict[str, Any]:
    """Record a tool call whose arguments did not decode into an object.

    Tool-availability and phase errors take precedence over argument-format errors.
    """
    error = _dispatch_gate_error(state, actor, tool_name)
    if error is None:
        error = error_string("InvalidArgumentError", "arguments are not a JSON object")
    return _protocol_rejection(state, actor, tool_name, error)


def _int_arg(arguments: dict[str, Any], key: str, default: int) -> int:
    value = arguments.get(key, default)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


ToolHandler = Callable[[dict[str, Any], str, dict[str, Any]], dict[str, Any]]

# Handler registry corresponding to the offered tool schemas.
TOOL_HANDLERS: dict[str, ToolHandler] = {
    "read_code": lambda state, actor, arguments: _read_code(state=state, actor=actor),
    "write_test_file": lambda state, actor, arguments: _write_test_file(
        state=state,
        actor=actor,
        filename=str(arguments.get("filename", "")),
        content=str(arguments.get("content", "")),
    ),
    "run_tests": lambda state, actor, arguments: _run_tests(
        state=state,
        actor=actor,
        test_file=str(arguments.get("test_file", "")),
        tool_name="run_tests",
    ),
    "read_source": lambda state, actor, arguments: _read_source(
        state=state,
        actor=actor,
    ),
    "resolve_records": lambda state, actor, arguments: _resolve_records(
        state=state,
        actor=actor,
        records=arguments.get("records"),
    ),
    "inspect_database": lambda state, actor, arguments: _inspect_database(
        state=state,
        actor=actor,
    ),
    "query_database": lambda state, actor, arguments: _query_database(
        state=state,
        actor=actor,
        sql=str(arguments.get("sql", "")),
        max_rows=_int_arg(arguments, "max_rows", 100),
    ),
    "save_final_answer": lambda state, actor, arguments: _save_final_answer(
        state,
        actor,
        arguments,
    ),
    "send_message": lambda state, actor, arguments: _send_message(
        state=state,
        actor=actor,
        content=str(arguments.get("content", "")),
        message_type=str(arguments.get("message_type", "other")),
    ),
    "get_log": lambda state, actor, arguments: _get_log(state=state, actor=actor),
    FINAL_VERDICT_TOOL_NAME: lambda state, actor, arguments: _submit_verdict(
        state=state,
        actor=actor,
        verdict=str(arguments.get("verdict", "")),
        reasoning=str(arguments.get("reasoning", "")),
    ),
}


def _execute_tool(
    state: dict[str, Any],
    actor: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Invoke a registered handler without protocol checks.

    An unregistered name raises KeyError, indicating a schema/handler mismatch.
    """
    return TOOL_HANDLERS[tool_name](state, actor, arguments)


def execute_tool_call(
    state: dict[str, Any],
    actor: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    error = _dispatch_gate_error(state, actor, tool_name)
    if error is not None:
        return _protocol_rejection(state, actor, tool_name, error)

    slot = agent_state(state, actor)
    result = _execute_tool(state, actor, tool_name, arguments)
    if tool_name not in _UNLOGGED_TOOLS:
        # Task-phase tools run before communication rounds begin.
        slot["private_raw_log"].append(
            {
                "seq": len(slot["private_raw_log"]) + 1,
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
            }
        )
    return result


def format_tool_content(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
