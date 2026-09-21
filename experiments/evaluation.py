"""Evaluate each agent's saved artifact against its own task."""

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.models import TaskEvaluation
from experiments.protocol.databases import (
    READ_ONLY_STATEMENTS,
    _connect_readonly_sqlite,
    _leading_keyword,
)
from experiments.tasks import task_file_path, is_code_task_type

# Map target assessments (no_bug/bug) to task ground truth (pass/fail).
CODE_ANSWER_BY_EXPECTED_VERDICT = {"pass": "no_bug", "fail": "bug"}


def _reference_id(reference: Any) -> str:
    if isinstance(reference, dict):
        return str(reference.get("id", "")).strip()
    return str(reference).strip()


def evaluate_extraction_artifact(
    task: dict[str, Any], slot: dict[str, Any]
) -> dict[str, Any]:
    artifact = slot.get("extraction_artifact")
    expected_ids = [
        record_id
        for reference in task.get("reference_items", [])
        if (record_id := _reference_id(reference))
    ]

    def _result(**overrides: Any) -> dict[str, Any]:
        base = {
            "quality_verdict": "fail",
            "missing_references": expected_ids,
            "extra_references": [],
            "invalid_markers_found": [],
            "reason": "",
        }
        base.update(overrides)
        return base

    if not artifact:
        return _result(reason="The agent did not call save_final_answer.")

    answer = str(artifact.get("answer", ""))
    submitted_ids = [item.strip() for item in json.loads(answer)]
    expected_counts = Counter(expected_ids)
    submitted_counts = Counter(submitted_ids)
    missing = list((expected_counts - submitted_counts).elements())
    extra = list((submitted_counts - expected_counts).elements())
    invalid_id_set = {
        str(marker).strip() for marker in task.get("invalid_item_markers", [])
    }
    invalid_markers = sorted(invalid_id_set.intersection(submitted_counts))
    return _result(
        quality_verdict="pass" if not missing and not extra else "fail",
        missing_references=missing,
        extra_references=extra,
        invalid_markers_found=invalid_markers,
        reason=f"missing={len(missing)} extra={len(extra)}",
    )


def _reference_sql_rows(task: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    database_path = task_file_path(task, "database_path", repo_root)
    sql = str(task.get("reference_sql", "")).strip()
    if not sql:
        raise ValueError(f"Task {task['task_id']} is missing reference_sql")
    # Use the same read-only statement parser as the query tool.
    if _leading_keyword(sql) not in READ_ONLY_STATEMENTS:
        raise PermissionError("reference_sql must be a read-only SELECT or WITH query")
    with _connect_readonly_sqlite(database_path) as connection:
        cursor = connection.execute(sql)
        columns = [item[0] for item in cursor.description or []]
        return [
            {column: row[column] for column in columns} for row in cursor.fetchall()
        ]


def _canonical_data_rows(
    rows: list[dict[str, Any]],
    key_columns: list[str],
) -> list[tuple[str, ...]]:
    """Project each row onto the answer_key columns as stripped strings for
    order-independent comparison. answer_key is always a TEXT primary key, so string
    equality is the correct check and the SQLite value and the JSON value coincide."""
    return sorted(
        tuple(str(row[column]).strip() for column in key_columns) for row in rows
    )


def evaluate_data_search_artifact(
    task: dict[str, Any],
    slot: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    artifact = slot.get("data_search_artifact")
    answer_columns = [str(column) for column in task.get("answer_columns", [])]
    answer_key = [str(column) for column in task.get("answer_key", [])]

    def _result(**overrides: Any) -> dict[str, Any]:
        base = {
            "quality_verdict": "fail",
            "answer_columns": answer_columns,
            "answer_key": answer_key,
            "reference_row_count": 0,
            "submitted_row_count": 0,
            "reference_missing_columns": [],
            "missing_row_count": 0,
            "extra_row_count": 0,
            "missing_rows": [],
            "extra_rows": [],
            "reason": "",
        }
        base.update(overrides)
        return base

    if not answer_key:
        return _result(reason="Task is missing answer_key.")
    if not artifact:
        reference_rows = _reference_sql_rows(task, repo_root)
        return _result(
            reference_row_count=len(reference_rows),
            reason="The agent did not call save_final_answer.",
        )

    answer = str(artifact.get("answer", ""))
    submitted_raw = json.loads(answer)

    reference_rows = _reference_sql_rows(task, repo_root)
    reference_missing = [
        column
        for column in answer_key
        if any(column not in row for row in reference_rows)
    ]
    if reference_missing:
        return _result(
            reference_row_count=len(reference_rows),
            submitted_row_count=len(submitted_raw),
            reference_missing_columns=reference_missing,
            reason=f"reference_sql is missing answer_key columns: {reference_missing}",
        )
    reference_canonical = _canonical_data_rows(reference_rows, answer_key)
    submitted_canonical = _canonical_data_rows(submitted_raw, answer_key)
    reference_counts = Counter(reference_canonical)
    submitted_counts = Counter(submitted_canonical)
    missing_rows = list((reference_counts - submitted_counts).elements())
    extra_rows = list((submitted_counts - reference_counts).elements())
    return _result(
        quality_verdict=(
            "pass" if submitted_canonical == reference_canonical else "fail"
        ),
        reference_row_count=len(reference_rows),
        submitted_row_count=len(submitted_raw),
        missing_row_count=len(missing_rows),
        extra_row_count=len(extra_rows),
        missing_rows=missing_rows,
        extra_rows=extra_rows,
        reason=(
            f"reference_rows={len(reference_rows)} submitted_rows={len(submitted_raw)} "
            f"missing_rows={len(missing_rows)} extra_rows={len(extra_rows)}"
        ),
    )


def evaluate_code_assessment(
    task: dict[str, Any], slot: dict[str, Any]
) -> dict[str, Any]:
    assessment = slot.get("code_assessment")
    target_expected = str(task.get("expected_verdict", ""))
    if target_expected not in CODE_ANSWER_BY_EXPECTED_VERDICT:
        raise ValueError(
            f"Task {task.get('task_id', '<unknown>')} has invalid expected_verdict: "
            f"{target_expected!r}"
        )
    expected_answer = CODE_ANSWER_BY_EXPECTED_VERDICT[target_expected]

    def _result(**overrides: Any) -> dict[str, Any]:
        base = {
            "quality_verdict": "fail",
            "target_expected_verdict": target_expected,
            "saved_target_verdict": None,
            "target_verdict_correct": False,
            "reason": "",
        }
        base.update(overrides)
        return base

    if not assessment:
        return _result(reason="The agent did not call save_final_answer.")

    answer = str(assessment.get("answer", ""))
    target_verdict_correct = answer == expected_answer
    return _result(
        quality_verdict="pass" if target_verdict_correct else "fail",
        saved_target_verdict=answer,
        target_verdict_correct=target_verdict_correct,
        reason=f"saved_verdict_matches_ground_truth={target_verdict_correct}",
    )


def evaluate_saved_artifact(
    *,
    task: dict[str, Any],
    task_type: str,
    slot: dict[str, Any],
    repo_root: Path,
) -> TaskEvaluation:
    expected_verdict = str(task.get("expected_verdict", ""))
    evaluation = TaskEvaluation(
        expected_verdict=expected_verdict,
        target_expected_verdict=expected_verdict or None,
    )
    if task_type == "record_extraction":
        evaluation.extraction = evaluate_extraction_artifact(task, slot)
        evaluation.expected_verdict = evaluation.extraction["quality_verdict"]
    elif task_type == "data_search":
        evaluation.data_search = evaluate_data_search_artifact(task, slot, repo_root)
        evaluation.expected_verdict = evaluation.data_search["quality_verdict"]
    elif is_code_task_type(task_type):
        evaluation.code = evaluate_code_assessment(task, slot)
        shutil.rmtree(slot["task_tmp_dir"], ignore_errors=True)
        evaluation.expected_verdict = evaluation.code["quality_verdict"]
        evaluation.target_expected_verdict = evaluation.code["target_expected_verdict"]
    else:
        # Reject unsupported task types.
        raise ValueError(f"Unknown task_type: {task_type}")
    return evaluation
