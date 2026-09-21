"""Typed, portable errors for agent-visible tool results.

Results use ``<kind>: <detail>`` with a kind from ERROR_KINDS. Bound detail lists
and remove local paths because errors become part of task evidence in raw logs.
Unexpected exceptions use the handler's fallback kind.
"""

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Error kinds grouped by dispatch gate and handler operation.
ERROR_KINDS: tuple[str, ...] = (
    # the dispatch gates
    "ToolUnavailableError",  # not a tool this task type has, in any phase
    "PhaseError",  # a tool of this task, called in the wrong phase
    "ProtocolError",  # the right phase, the wrong step within it
    # what a handler rejects
    "InvalidArgumentError",  # an argument the tool cannot use
    "InvalidAnswerFormatError",  # save_final_answer's answer, against its task type
    "PermissionError",  # the operation itself is not allowed
    # what a handler attempted and could not finish
    "TaskFileError",  # reading this agent's own task file or workspace
    "QueryError",  # running a query against this agent's database
    "TestRunError",  # writing, compiling or running a test
    # the two channel tools
    "MessageError",  # send_message
    "VerdictError",  # submit_verdict
)


class UnknownErrorKind(BaseException):
    """Signal an invalid error kind outside normal tool-error handling.

    Inherit from BaseException so handler-level ``except Exception`` cannot hide it.
    """


class ToolError(Exception):
    """A failure whose kind the handler naming it has already decided."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(error_string(kind, detail))
        self.kind = kind
        self.detail = detail


def error_string(kind: str, detail: str) -> str:
    """Format a tool failure after validating its error kind."""
    if kind not in ERROR_KINDS:
        raise UnknownErrorKind(f"Unknown error kind: {kind!r}")
    return f"{kind}: {detail}"


# How many offending positions one detail names before it stops counting them.
MAX_LISTED_POSITIONS = 10


def listed_positions(items: Sequence[Any], separator: str = ", ") -> str:
    """Summarize the first few invalid entries and count the remainder."""
    listed = separator.join(str(item) for item in items[:MAX_LISTED_POSITIONS])
    remaining = len(items) - MAX_LISTED_POSITIONS
    if remaining > 0:
        return f"{listed}{separator}and {remaining} more"
    return listed


def portable_detail(detail: str, location: Any = "") -> str:
    """Remove a known local directory prefix from error text."""
    root = str(location or "").rstrip("/")
    if not root:
        return detail
    return detail.replace(f"{root}/", "").replace(root, "")


# Match absolute paths with at least two segments, preserving traceback suffixes
# and avoiding URL separators.
_ABSOLUTE_PATH = re.compile(r"(?<![\w/])/(?:[\w.+-]+/)+[\w.+-]+")


def without_local_paths(detail: str) -> str:
    """Reduce absolute paths to filenames and remove the home-directory name."""
    home = str(Path.home()).rstrip("/")

    def reduced(match: re.Match[str]) -> str:
        matched = match.group(0)
        return "" if matched == home else matched.rsplit("/", 1)[-1]

    # Also remove single-segment home paths such as /root.
    return portable_detail(_ABSOLUTE_PATH.sub(reduced, detail), home)


def tool_error_string(
    exc: BaseException,
    *,
    fallback_kind: str,
    repo_root: Any = "",
) -> str:
    """Translate a caught exception into a tool error using the handler's fallback kind."""
    if isinstance(exc, ToolError):
        kind, detail = exc.kind, exc.detail
    elif isinstance(exc, subprocess.TimeoutExpired):
        # Avoid str(exc), which includes the full command and absolute paths.
        kind, detail = "TestRunError", f"test run exceeded {exc.timeout:g}s"
    elif isinstance(exc, OSError):
        # Keep the handler's error kind and omit the local filename.
        reason = exc.strerror or str(exc)
        name = Path(exc.filename).name if exc.filename else ""
        kind = fallback_kind
        detail = f"{reason}: {name}" if name else str(reason)
    else:
        kind, detail = fallback_kind, str(exc)
    return error_string(kind, without_local_paths(portable_detail(detail, repo_root)))
