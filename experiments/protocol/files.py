"""Read/write file tools for task artifacts and generated tests."""

from pathlib import Path
from typing import Any, Callable

from experiments.protocol.errors import (
    ToolError,
    tool_error_string,
)
from experiments.protocol.paths import (
    _resolve_task_file_path,
    _resolve_test_file,
)
from experiments.protocol.records import strip_record_markers
from experiments.protocol.state import _log_event, agent_state

# Share the test filename pattern with the tool schema.
TEST_FILENAME_PATTERN = "test_*.py"


def _test_filename_error(filename: str) -> str | None:
    prefix, suffix = TEST_FILENAME_PATTERN.split("*")
    if filename.startswith(prefix) and filename.endswith(suffix):
        return None
    return f"test filename for this task's language must match {TEST_FILENAME_PATTERN}"


def _read_task_file(
    state: dict[str, Any],
    actor: str,
    slot_key: str,
    tool_name: str,
    transform: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    slot = agent_state(state, actor)
    try:
        resolved = _resolve_task_file_path(state, slot, slot_key, tool_name)
        content = resolved.read_text(encoding="utf-8")
        if transform is not None:
            content = transform(content)
        # Omit local paths from agent-visible results and raw logs.
        result = {"success": True, "content": content}
    except Exception as exc:
        result = {
            "success": False,
            "error": tool_error_string(
                exc,
                fallback_kind="TaskFileError",
                repo_root=state.get("repo_root", ""),
            ),
        }
    _log_event(
        state,
        {
            "actor": actor,
            "tool": tool_name,
            "success": result["success"],
            **({"error": result["error"]} if not result["success"] else {}),
        },
    )
    return result


def _read_source(state: dict[str, Any], actor: str) -> dict[str, Any]:
    transform = (
        strip_record_markers
        if state.get("task_type") == "record_extraction"
        else None
    )
    return _read_task_file(
        state,
        actor,
        "source_path",
        "read_source",
        transform=transform,
    )


def _read_code(state: dict[str, Any], actor: str) -> dict[str, Any]:
    return _read_task_file(state, actor, "target_code_path", "read_code")


def _write_test_file(
    state: dict[str, Any], actor: str, filename: str, content: str
) -> dict[str, Any]:
    slot = agent_state(state, actor)
    written: Path | None = None
    try:
        safe_name = Path(filename).name
        name_error = _test_filename_error(safe_name)
        if name_error is not None:
            raise ToolError("InvalidArgumentError", name_error)
        path = _resolve_test_file(slot, safe_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        slot["written_test_files"].append(str(path))
        written = path
        # Return a relative filename; absolute scratch paths contain internal task IDs.
        result = {"success": True, "path": path.name}
    except Exception as exc:
        result = {
            "success": False,
            "error": tool_error_string(
                exc,
                fallback_kind="TestRunError",
                repo_root=state.get("repo_root", ""),
            ),
        }
    _log_event(
        state,
        {
            "actor": actor,
            "tool": "write_test_file",
            "success": result["success"],
            "path": str(written or filename),
            **({"error": result["error"]} if not result["success"] else {}),
        },
    )
    return result
