"""Resolve task-local paths within the repository and each agent's workspace.

Resource files come from the caller's task slot. User-supplied test filenames
are additionally restricted to that task's scratch directory.
"""

from pathlib import Path
from typing import Any

from experiments.protocol.errors import ToolError


def _repo_root(state: dict[str, Any]) -> Path:
    root = state.get("repo_root")
    if not root:
        raise ToolError("TaskFileError", "repo_root is not configured")
    return Path(root).resolve()


def _task_tmp_dir(slot: dict[str, Any]) -> Path:
    tmp_dir = slot.get("task_tmp_dir")
    if not tmp_dir:
        raise ToolError("TaskFileError", "task_tmp_dir is not configured")
    path = Path(tmp_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_task_file_path(
    state: dict[str, Any],
    slot: dict[str, Any],
    slot_key: str,
    label: str,
) -> Path:
    repo_root = _repo_root(state)
    configured = str(slot.get(slot_key, "")).strip()
    if not configured:
        # Reject empty paths, which would otherwise resolve to the repository root.
        raise ToolError("TaskFileError", f"{slot_key} is not configured")
    path = Path(configured)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.is_relative_to(repo_root):
        raise ToolError("PermissionError", f"{label} path must stay inside repo")
    return path


def _resolve_test_file(slot: dict[str, Any], test_file: str) -> Path:
    tmp_dir = _task_tmp_dir(slot)
    path = Path(test_file)
    if not path.name:
        # Reject empty or dot filenames, which would resolve to the scratch directory.
        raise ToolError("InvalidArgumentError", "test_file must name a file")
    if not path.is_absolute():
        path = (tmp_dir / path).resolve()
    else:
        path = path.resolve()
    if not path.is_relative_to(tmp_dir):
        raise ToolError(
            "PermissionError", "test file must stay inside task temp directory"
        )
    return path
