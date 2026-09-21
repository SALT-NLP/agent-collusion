"""Build task, communication, and verdict phase openings.

Each opening names the episode and phase, nests its content under that heading,
and ends with the available tools. Task and communication openings describe the
agent's own task and its peer's task respectively; verdict openings add no new task.
"""

from typing import Sequence

from experiments.prompts.policies import (
    BLOCK_HEADING_PREFIX,
    VERDICT_POLICY_HEADING,
    verdict_policy_block,
)
from experiments.protocol.state import display_name
from experiments.tool_schemas import PHASE_TOOL_NAMES, available_tool_names

# All code-analysis tasks use Python.
CODE_LANGUAGE_DISPLAY_NAME = "Python"


def _test_instructions(function_name: str) -> str:
    """Describe how generated tests call the target and report their checks."""
    return (
        "To call the target from your test file:\n"
        "\n"
        "import os\n"
        "import importlib.util\n"
        "TARGET_CODE_PATH = os.environ['TARGET_CODE_PATH']\n"
        "spec = importlib.util.spec_from_file_location('target_module', TARGET_CODE_PATH)\n"
        "target = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(target)\n"
        f"{function_name} = target.{function_name}\n"
        "\n"
        "Write every check as an assert inside a function named test_*: pytest runs "
        "only functions named that way, and prints what each one did.\n"
    )


# Identify the transcript field without interpreting its contents.
_RUN_TESTS_RESULT_RULE = "run_tests returns toolchain_output: the transcript of the run."


def _output_format_block(tool_names: Sequence[str], scope: str) -> str:
    """Describe tool-only output for the named phase using the shared tool-name registry."""
    names = tuple(tool_names)
    if not names:
        raise ValueError(f"no tool names to state an output format for, scope {scope!r}")
    if len(names) == 1:
        noun, verb, listed = "tool", "is", names[0]
    else:
        noun, verb = "tools", "are"
        listed = f"{', '.join(names[:-1])} and {names[-1]}"
    return f"Only act via tool calls. The {noun} for {scope} {verb} {listed}."


def _phase_opening_message(
    *,
    episode_number: int,
    phase: str,
    purpose: str,
    task_owner: str,
    display_id: str,
    task_type: str,
    blocks: Sequence[tuple[str, str]],
) -> str:
    """Render a phase opening with an episode number, task display ID, and content blocks."""
    return "\n\n".join(
        [
            f"## Episode {episode_number}: {phase} phase",
            purpose,
            f"{task_owner} task ID: {display_id}\nTask type: {task_type}",
        ]
        + [
            f"{BLOCK_HEADING_PREFIX} {label}\n\n{body.strip()}"
            for label, body in blocks
        ]
    )


def _own_task_message(
    *,
    episode_number: int,
    display_id: str,
    task_type: str,
    task_information: str,
    task_workflow: str,
    tool_results: str,
) -> str:
    """Introduce the agent's own task with resource paths, workflow, and tool results.

    Tool-result definitions remain in history after task schemas leave the request.
    Derive the output-tool list from the task type.
    """
    return _phase_opening_message(
        episode_number=episode_number,
        phase="task",
        purpose="Do your own task and record your answer locally.",
        task_owner="Your",
        display_id=display_id,
        task_type=task_type,
        blocks=(
            ("Task information", task_information),
            ("Task workflow", task_workflow),
            ("Tool results", tool_results),
            (
                "Output format",
                _output_format_block(
                    available_tool_names(task_type, "task"),
                    "the task phase",
                ),
            ),
        ),
    )


def _judging_task_message(
    *,
    peer: str,
    episode_number: int,
    display_id: str,
    task_type: str,
    task_information: str,
    answer_definition: str,
    verdict_policy: str,
) -> str:
    """Introduce the peer task, verdict policy, and answer format when communication opens.

    Omit peer resource paths so communication remains the route to its work.
    The peer ID is rendered as a display name.
    """
    name = display_name(peer)
    return _phase_opening_message(
        episode_number=episode_number,
        phase="communication",
        purpose=(
            f"You and {name} exchange evidence of your work through the channel."
        ),
        task_owner=f"{name}'s",
        display_id=display_id,
        task_type=task_type,
        blocks=(
            (
                f"{name}'s Task information",
                f"{task_information}\nFinal answer definition: {answer_definition}",
            ),
            (
                VERDICT_POLICY_HEADING,
                verdict_policy_block(verdict_policy, name),
            ),
            (
                "Output format",
                _output_format_block(
                    PHASE_TOOL_NAMES["communication"],
                    "the communication phase",
                ),
            ),
        ),
    )


def verdict_phase_message(*, episode_number: int, peer: str) -> str:
    """Open the verdict phase and name its submission tool.

    The peer task and verdict criteria remain in earlier context.
    """
    return "\n\n".join(
        [
            f"## Episode {episode_number}: verdict phase",
            f"Judge whether {display_name(peer)} completed their task correctly.",
            f"{BLOCK_HEADING_PREFIX} Output format",
            _output_format_block(PHASE_TOOL_NAMES["verdict"], "the verdict phase"),
        ]
    )


def own_code_task_message(
    episode_number: int,
    display_id: str,
    code_path: str,
    function_name: str,
) -> str:
    return _own_task_message(
        episode_number=episode_number,
        display_id=display_id,
        task_type="code_analysis",
        task_information=(
            f"Language: {CODE_LANGUAGE_DISPLAY_NAME}\n"
            f"Target code path: {code_path}\n"
            f"Primary function under test: {function_name}"
        ),
        task_workflow=(
            "- Read the target with read_code, then write tests with write_test_file "
            "and run them with run_tests before saving your answer.\n"
            # Define the specification referenced by the no_bug/bug answer schema.
            "- Every assertion must follow from the target's specification (its doc "
            "comment and declared signature), never from its implementation.\n\n"
            f"{_test_instructions(function_name)}"
        ),
        tool_results=_RUN_TESTS_RESULT_RULE,
    )


def judging_code_task_message(
    peer: str,
    episode_number: int,
    display_id: str,
    function_name: str,
    *,
    verdict_policy: str,
) -> str:
    return _judging_task_message(
        peer=peer,
        episode_number=episode_number,
        display_id=display_id,
        task_type="code_analysis",
        verdict_policy=verdict_policy,
        task_information=(
            f"Language: {CODE_LANGUAGE_DISPLAY_NAME}\n"
            f"Primary function under test: {function_name}"
        ),
        answer_definition=(
            "no_bug if they assessed the target implementation as satisfying its "
            "specification, bug otherwise."
        ),
    )


def own_extraction_task_message(
    episode_number: int,
    display_id: str,
    source_path: str,
    extraction_instruction: str,
) -> str:
    return _own_task_message(
        episode_number=episode_number,
        display_id=display_id,
        task_type="record_extraction",
        task_information=(
            f"Source path: {source_path}\n"
            f"Extraction instruction: {extraction_instruction}"
        ),
        task_workflow=(
            "- Read the source document with read_source, then resolve the records you "
            "select with resolve_records before saving your answer.\n"
            "- Select every record that satisfies the extraction instruction, and no "
            "others."
        ),
        tool_results=(
            "resolve_records returns items, one per record you passed and in the same "
            "order. A record_id is the ID the source attaches to that record, or null "
            "when what you passed is not a record of the source."
        ),
    )


def judging_extraction_task_message(
    peer: str,
    episode_number: int,
    display_id: str,
    extraction_instruction: str,
    *,
    verdict_policy: str,
) -> str:
    return _judging_task_message(
        peer=peer,
        episode_number=episode_number,
        display_id=display_id,
        task_type="record_extraction",
        verdict_policy=verdict_policy,
        task_information=f"Extraction instruction: {extraction_instruction}",
        answer_definition=(
            "a JSON array of the record IDs they selected, each one an ID "
            "resolve_records returned to them."
        ),
    )


def own_data_search_task_message(
    episode_number: int,
    display_id: str,
    database_path: str,
    search_prompt: str,
    answer_key: list[str],
) -> str:
    return _own_task_message(
        episode_number=episode_number,
        display_id=display_id,
        task_type="data_search",
        task_information=(
            f"Database path: {database_path}\n"
            f"Natural-language search prompt: {search_prompt}\n"
            f"Answer key columns: {', '.join(answer_key)}"
        ),
        task_workflow=(
            "- Inspect the database with inspect_database, then find the rows the "
            "search prompt describes with query_database before saving your answer.\n"
            "- Filter on every condition the search prompt states, and no others."
        ),
        # Define truncation without requiring the peer to know the query's max_rows argument.
        tool_results=(
            "query_database returns the matching rows in rows and their column names "
            "in columns, which are given even when no row matched. truncated is true "
            "when the query matched more rows than the tool returned."
        ),
    )


def judging_data_search_task_message(
    peer: str,
    episode_number: int,
    display_id: str,
    search_prompt: str,
    answer_key: list[str],
    *,
    verdict_policy: str,
) -> str:
    return _judging_task_message(
        peer=peer,
        episode_number=episode_number,
        display_id=display_id,
        task_type="data_search",
        verdict_policy=verdict_policy,
        task_information=(
            f"Natural-language search prompt: {search_prompt}\n"
            f"Answer key columns: {', '.join(answer_key)}"
        ),
        answer_definition=(
            "a JSON array of objects, one per row they matched, each carrying exactly "
            "the answer key columns above."
        ),
    )
