"""Test execution helpers for code-analysis tasks."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.protocol.errors import (
    ToolError,
    portable_detail,
    tool_error_string,
    without_local_paths,
)
from experiments.protocol.paths import (
    _task_tmp_dir,
    _repo_root,
    _resolve_test_file,
)
from experiments.protocol.state import _log_event, agent_state


# Restrict the test environment to avoid exposing provider credentials.
# Keep PATH for toolchain lookup.
_INHERITED_ENV_VARS = ("PATH", "TMPDIR", "LANG", "LC_ALL")


def _toolchain_env(tmp_dir: Path, **extra: str) -> dict[str, str]:
    """Build the test environment with HOME set to the task's scratch directory."""
    env = {
        name: os.environ[name] for name in _INHERITED_ENV_VARS if name in os.environ
    }
    env["HOME"] = str(tmp_dir)
    env.update(extra)
    return env


# One test run's wall-clock bound.
_PROCESS_TIMEOUT = 30


def _run_process(
    command: list[str], cwd: Any, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run the test process with stderr merged into stdout and stdin closed."""
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=_PROCESS_TIMEOUT,
        check=False,
    )


# Keep absolute commands in event logs only; agents receive the scrubbed transcript.
_LOG_ONLY_FIELDS = ("command",)


def _agent_visible_result(
    result: dict[str, Any], slot: dict[str, Any], repo_root: Any
) -> dict[str, Any]:
    """Prepare agent-visible test output, removing command metadata and local paths.

    Replace task-directory paths before generic paths to hide internal task IDs.
    """
    visible = {
        key: value for key, value in result.items() if key not in _LOG_ONLY_FIELDS
    }
    output = str(visible.get("toolchain_output", ""))
    if output:
        root = Path(str(repo_root)).resolve() if repo_root else None
        tmp_dir = str(slot.get("task_tmp_dir", "") or "")
        if tmp_dir:
            absolute = Path(tmp_dir).resolve()
            output = portable_detail(output, absolute)
            if root is not None and absolute.is_relative_to(root):
                output = portable_detail(output, absolute.relative_to(root))
        output = portable_detail(output, repo_root)
        output = without_local_paths(output)
        visible["toolchain_output"] = output
    return visible


def _run_python_tests(
    state: dict[str, Any], slot: dict[str, Any], path: Path
) -> dict[str, Any]:
    repo_root = _repo_root(state)
    pytest_executable = shutil.which("pytest")
    if not pytest_executable and importlib.util.find_spec("pytest") is None:
        # Fail on missing toolchains before their startup errors can be read as test evidence.
        raise ToolError(
            "TestRunError", "pytest not found (pip install -r requirements.txt)"
        )
    env = _toolchain_env(
        _task_tmp_dir(slot),
        TARGET_CODE_PATH=str(Path(slot["target_code_path"]).resolve()),
        PYTHONDONTWRITEBYTECODE="1",
    )
    launcher = (
        [pytest_executable] if pytest_executable else [sys.executable, "-m", "pytest"]
    )
    # Suppress environment metadata in the test banner to keep raw logs task-focused.
    command = [*launcher, str(path), "-vv", "--no-header", "-p", "no:cacheprovider"]
    completed = _run_process(command, cwd=repo_root, env=env)
    return {
        "success": True,
        "command": " ".join(command),
        "toolchain_output": completed.stdout,
    }


def _reject_vacuous_test(slot: dict[str, Any], path: Path) -> None:
    """Reject a test file that does not reference the target function."""
    function_name = str(slot.get("function_name", ""))
    if not function_name:
        return
    source = path.read_text(encoding="utf-8", errors="replace")
    if function_name not in source:
        raise ToolError(
            "TestRunError",
            f"test file {path.name} never references {function_name}, "
            "so it cannot exercise the target code",
        )


def _log_run_tests_event(
    state: dict[str, Any],
    slot: dict[str, Any],
    actor: str,
    tool_name: str,
    result: dict[str, Any],
) -> None:
    """Record a test invocation with its absolute command for offline analysis."""
    _log_event(
        state,
        {
            "actor": actor,
            "tool": tool_name,
            "success": result["success"],
            "command": result.get("command"),
            **({"error": result["error"]} if not result["success"] else {}),
        },
    )


def _run_tests(
    state: dict[str, Any], actor: str, test_file: str, tool_name: str = "run_tests"
) -> dict[str, Any]:
    slot = agent_state(state, actor)
    try:
        path = _resolve_test_file(slot, test_file)
        if not path.exists():
            raise ToolError("TestRunError", f"test file does not exist: {path.name}")
        _reject_vacuous_test(slot, path)
        result = _run_python_tests(state, slot, path)
    except Exception as exc:
        result = {
            "success": False,
            "error": tool_error_string(
                exc,
                fallback_kind="TestRunError",
                repo_root=state.get("repo_root", ""),
            ),
        }
    # Log the full result before removing command metadata from agent-visible output.
    _log_run_tests_event(state, slot, actor, tool_name, result)
    return _agent_visible_result(result, slot, state.get("repo_root", ""))
