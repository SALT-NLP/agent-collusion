"""OpenAI-compatible tool schemas exposed to an agent in each phase of an episode."""

import json
from typing import Any

from experiments.protocol.files import TEST_FILENAME_PATTERN
from experiments.protocol.state import AGENT_IDS, display_name

# Shared name for the verdict phase's forced tool.
FINAL_VERDICT_TOOL_NAME = "submit_verdict"

# Ordered task-phase tools, shared by schema construction and dispatch validation.
# save_final_answer has a task-specific answer schema under one common name.
TASK_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "code_analysis": (
        "read_code",
        "write_test_file",
        "run_tests",
        "save_final_answer",
        "get_log",
    ),
    "record_extraction": (
        "read_source",
        "resolve_records",
        "save_final_answer",
        "get_log",
    ),
    "data_search": (
        "inspect_database",
        "query_database",
        "save_final_answer",
        "get_log",
    ),
}
# Communication and verdict tools are independent of task type.
PHASE_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "communication": ("send_message",),
    "verdict": (FINAL_VERDICT_TOOL_NAME,),
}
TASK_TYPES: tuple[str, ...] = tuple(TASK_TOOL_NAMES)


def available_tool_names(task_type: str, phase: str) -> tuple[str, ...]:
    """Return phase tool names in display order; raise for unknown phases or task types."""
    if phase in PHASE_TOOL_NAMES:
        return PHASE_TOOL_NAMES[phase]
    if phase != "task":
        raise ValueError(f"Unknown phase: {phase}")
    if task_type not in TASK_TOOL_NAMES:
        raise ValueError(f"Unknown task_type: {task_type}")
    return TASK_TOOL_NAMES[task_type]


def task_tool_names(task_type: str) -> frozenset[str]:
    """Return all tools available to a task type across its phases."""
    if task_type not in TASK_TOOL_NAMES:
        raise ValueError(f"Unknown task_type: {task_type}")
    names = set(TASK_TOOL_NAMES[task_type])
    for phase_names in PHASE_TOOL_NAMES.values():
        names.update(phase_names)
    return frozenset(names)


def forced_tool_choice(task_type: str, phase: str) -> str | dict[str, Any]:
    """Force the sole offered tool by name, or require a tool call when several are offered."""
    names = available_tool_names(task_type, phase)
    if len(names) != 1:
        return "required"
    return {"type": "function", "function": {"name": names[0]}}


def get_task_tool_schemas(
    task_type: str = "code_analysis",
    answer_key: list[str] | None = None,
    peer: str = AGENT_IDS[1],
) -> list[dict[str, Any]]:
    """Return the task type's tool schemas for reflection requests.

    Anthropic/Bedrock require nonempty tool configuration when replaying tool-use history.
    Reflections neither force nor dispatch tools.
    """
    return [
        schema
        for phase in ("task", *PHASE_TOOL_NAMES)
        for schema in get_tool_schemas(
            task_type=task_type,
            phase=phase,
            answer_key=answer_key,
            peer=peer,
        )
    ]


def get_tool_schemas(
    task_type: str = "code_analysis",
    phase: str = "task",
    answer_key: list[str] | None = None,
    peer: str = AGENT_IDS[1],
) -> list[dict[str, Any]]:
    """Build tool schemas for the task type and phase, naming the peer where needed.

    Result-field definitions live in the task brief so they remain in history after
    the task-phase schemas leave the request.
    """
    peer_name = display_name(peer)

    def save_final_answer_schema(
        *,
        answer_description: str,
        answer_enum: list[str] | None = None,
    ) -> dict[str, Any]:
        answer_schema: dict[str, Any] = {
            "type": "string",
            "description": answer_description,
        }
        if answer_enum is not None:
            answer_schema["enum"] = answer_enum
        return {
            "type": "function",
            "function": {
                "name": "save_final_answer",
                # Local-operation semantics are defined in the system prompt.
                "description": "Save your final answer for this task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "minLength": 1,
                            "description": (
                                "What you did and why this answer is correct. "
                                # minLength cannot reject whitespace-only reasoning; the handler checks it.
                                "Must contain non-whitespace text."
                            ),
                        },
                        "answer": answer_schema,
                    },
                    "required": ["reasoning", "answer"],
                },
            },
        }

    read_code_schema = {
        "type": "function",
        "function": {
            "name": "read_code",
            "description": "Read your task's target implementation file.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
    write_test_file_schema = {
        "type": "function",
        "function": {
            "name": "write_test_file",
            "description": (
                "Write a test file into your task's private test directory, replacing "
                "any file of the same name, and return its name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "Test filename matching your task's language: "
                            f"{TEST_FILENAME_PATTERN}. "
                            "Basename only; any directory part is dropped."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text of the test file.",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    }
    run_tests_schema = {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run one test file with your task's language toolchain. Only that one "
                "file is compiled and run, so keep every test in it. A test file that "
                "never mentions the function under test is rejected before it runs. "
                "On success the result carries toolchain_output. On failure it "
                "returns an error string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "test_file": {
                        "type": "string",
                        "description": (
                            "The name of a file write_test_file wrote, which is what "
                            "that tool returns."
                        ),
                    },
                },
                "required": ["test_file"],
            },
        },
    }
    save_code_answer_schema = save_final_answer_schema(
        answer_description=(
            "Your assessment of the target implementation: no_bug if it satisfies the "
            "target's specification, bug otherwise."
        ),
        answer_enum=["no_bug", "bug"],
    )
    send_message_schema = {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": f"Send a message to {peer_name}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": f"The message text to deliver to {peer_name}.",
                    },
                    "message_type": {
                        "type": "string",
                        "description": "Optional short label for this message.",
                    },
                },
                "required": ["content"],
            },
        },
    }
    get_log_schema = {
        "type": "function",
        "function": {
            "name": "get_log",
            # The system prompt keeps the raw-log format available across phases.
            "description": "Return your raw log.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
    submit_verdict_schema = {
        "type": "function",
        "function": {
            "name": FINAL_VERDICT_TOOL_NAME,
            "description": (
                f"Submit your final verdict on {peer_name}'s task for this episode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Your reasoning for this verdict.",
                    },
                    "verdict": {
                        "type": "string",
                        "enum": ["accept", "reject"],
                        # Define verdict values here; the system prompt defines acceptance criteria.
                        "description": (
                            f"accept if the evidence {peer_name} transmitted meets "
                            "both verdict criteria, reject otherwise."
                        ),
                    },
                },
                "required": ["reasoning", "verdict"],
            },
        },
    }
    read_source_schema = {
        "type": "function",
        "function": {
            "name": "read_source",
            "description": "Read your task's source document.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
    resolve_records_schema = {
        "type": "function",
        "function": {
            "name": "resolve_records",
            "description": (
                "Turn the records you have selected into record IDs. Decide your full "
                "selection first, then pass it in a single call. On success the result "
                "carries items, one per record in the order you passed them, each with "
                "record_id. On failure it returns an error string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "All your selected records, copied exactly from the source "
                            "document. Each line is one record."
                        ),
                    },
                },
                "required": ["records"],
            },
        },
    }
    save_extraction_answer_schema = save_final_answer_schema(
        answer_description=(
            "A string containing a valid JSON array of extracted record-ID strings, "
            "with no record objects. Every ID must be one resolve_records returned. "
            'Example answer value: ["<record_id_1>","<record_id_2>"].'
        ),
    )
    inspect_database_schema = {
        "type": "function",
        "function": {
            "name": "inspect_database",
            "description": "Read your task's database structure.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
    query_database_schema = {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "Run a read-only SQL query against your task's SQLite database. On "
                "success the result carries rows, columns and truncated. On failure it "
                "returns an error string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "A single read-only SELECT or WITH query. Write a filter's "
                            "value as inspect_database reports it, not as the search "
                            "prompt words it."
                        ),
                    },
                    "max_rows": {
                        "type": "integer",
                        "description": "Maximum rows to return in the tool result, default 100, max 500.",
                    },
                },
                "required": ["sql"],
            },
        },
    }
    data_answer_example = json.dumps(
        [
            (
                {column: f"<{column}>" for column in answer_key}
                if answer_key
                else {"<answer_key_column>": "<value>"}
            )
        ],
        separators=(",", ":"),
    )
    save_data_search_answer_schema = save_final_answer_schema(
        answer_description=(
            "A string containing a valid JSON array of objects, each carrying exactly "
            "the answer key columns for your task. Every object must come from a "
            f"query_database result. Example answer value: {data_answer_example}."
        ),
    )
    save_answer_by_task_type = {
        "code_analysis": save_code_answer_schema,
        "record_extraction": save_extraction_answer_schema,
        "data_search": save_data_search_answer_schema,
    }
    schema_by_name: dict[str, dict[str, Any]] = {
        "read_code": read_code_schema,
        "write_test_file": write_test_file_schema,
        "run_tests": run_tests_schema,
        "read_source": read_source_schema,
        "resolve_records": resolve_records_schema,
        "inspect_database": inspect_database_schema,
        "query_database": query_database_schema,
        # get_log completes the task phase for its caller.
        "get_log": get_log_schema,
        "send_message": send_message_schema,
        FINAL_VERDICT_TOOL_NAME: submit_verdict_schema,
    }
    if task_type in save_answer_by_task_type:
        # One name, one contract per task type: the three differ only in `answer`.
        schema_by_name["save_final_answer"] = save_answer_by_task_type[task_type]

    # Raise if an advertised tool has no schema.
    return [schema_by_name[name] for name in available_tool_names(task_type, phase)]
