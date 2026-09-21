"""Resolve extraction record text to IDs.

read_source removes ID markers; resolve_records maps the visible text back to IDs.
"""

import re
from typing import Any

from experiments.protocol.errors import (
    ToolError,
    listed_positions,
    tool_error_string,
)
from experiments.protocol.paths import _resolve_task_file_path
from experiments.protocol.state import _log_event, agent_state

# e.g. "[REC01-a3f9c2] Maya wrote in the apartment chat that ...".
RECORD_MARKER = re.compile(r"^\[([A-Za-z0-9]+-[A-Za-z0-9]+)\]\s*")


def strip_record_markers(content: str) -> str:
    """The source as an agent sees it: record text with no IDs attached."""
    return "\n".join(RECORD_MARKER.sub("", line) for line in content.split("\n"))


def _record_index(content: str) -> dict[str, str]:
    """Map each record's text onto its ID, keyed on exactly what the agent reads."""
    index: dict[str, str] = {}
    for line in content.split("\n"):
        match = RECORD_MARKER.match(line)
        if not match:
            continue
        text = line[match.end() :].strip()
        if text:
            index[text] = match.group(1)
    return index


def _records_error(records: Any) -> str | None:
    if not isinstance(records, list):
        return "records must be an array of strings"
    if not records:
        return "records must contain at least one record"
    malformed = [
        index
        for index, record in enumerate(records)
        if not isinstance(record, str) or not record.strip()
    ]
    if malformed:
        return (
            "every record must be a non-empty string; invalid indexes: "
            f"{listed_positions(malformed)}"
        )
    return None


def _resolve_records(state: dict[str, Any], actor: str, records: Any) -> dict[str, Any]:
    slot = agent_state(state, actor)
    try:
        error = _records_error(records)
        if error is not None:
            raise ToolError("InvalidArgumentError", error)
        # Task tools do not require other task tools to have run first.
        resolved_path = _resolve_task_file_path(
            state,
            slot,
            "source_path",
            "resolve_records",
        )
        index = _record_index(resolved_path.read_text(encoding="utf-8"))

        # Return IDs only; positions and match counts are derivable from the list.
        items = [{"record_id": index.get(record.strip())} for record in records]
        result = {"success": True, "items": items}
    except Exception as exc:
        result = {
            "success": False,
            "error": tool_error_string(
                exc,
                fallback_kind="TaskFileError",
                repo_root=state.get("repo_root", ""),
            ),
        }

    logged_items = result.get("items", [])
    resolved_ids = [item["record_id"] for item in logged_items if item["record_id"]]
    _log_event(
        state,
        {
            "actor": actor,
            "tool": "resolve_records",
            "success": result["success"],
            "record_count": len(records) if isinstance(records, list) else 0,
            "resolved_count": len(resolved_ids),
            "unmatched_count": len(logged_items) - len(resolved_ids),
            "resolved_ids": resolved_ids,
            **({"error": result["error"]} if not result["success"] else {}),
        },
    )
    return result
