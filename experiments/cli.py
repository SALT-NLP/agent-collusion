"""Build and launch repeated experiment runs from one explicit setting."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from experiments.protocol import DEFAULT_MAX_ROUNDS
from experiments.protocol.state import AGENT_IDS
from experiments.protocol.rewards import reward_scheme_from_args
from experiments.runner import (
    MAX_ROUNDS_HELP,
    add_memory_arguments,
    add_model_arguments,
    add_reward_scheme_arguments,
    add_feedback_arguments,
    replays_controlled_bob,
    validate_run_args,
)
from experiments.tasks import read_task_pairs

DEFAULT_VERDICT_POLICY = "raw-only"
MANIFEST_FILENAME_PATTERN = re.compile(r"rep(\d+)_sampled_manifest\.json")
SEQUENCE_DIR_PREFIX = "task_sequences_"
MERGED_SEQUENCE_RECORD = "merged_sequence_record.json"


def _setting_label(
    repeat_idx: int,
    verdict_policy_label: str,
    max_rounds: int,
) -> str:
    # Include the round count to distinguish sweep conditions.
    return f"default_cross-episode_{verdict_policy_label}_r{max_rounds}_rep{repeat_idx}"


def _resolve_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _load_task_sequence_manifests(
    records: list[str],
    repo_root: Path,
) -> list[Path]:
    """Resolve one manifest per repeat, concatenating records in the supplied order.

    Write and validate merged sequences before launching child runs.
    """
    resolved = [_resolve_path(record, repo_root) for record in records]
    per_record = [_sequence_record_manifests(path, repo_root) for path in resolved]
    if len(per_record) == 1:
        return per_record[0]
    return _write_merged_sequences(resolved, per_record, repo_root)


def _sequence_record_manifests(record_path: Path, repo_root: Path) -> list[Path]:
    """The per-repeat manifest paths one record holds, in repeat order."""
    if record_path.is_dir():
        indexed_manifests = []
        for path in record_path.glob("rep*_sampled_manifest.json"):
            match = MANIFEST_FILENAME_PATTERN.fullmatch(path.name)
            if match:
                indexed_manifests.append((int(match.group(1)), path))
        if not indexed_manifests:
            raise FileNotFoundError(
                f"No rep*_sampled_manifest.json files found in {record_path}"
            )
        indexed_manifests.sort(key=lambda item: item[0])
        _validate_contiguous_indexes(
            [index for index, _ in indexed_manifests],
            "--task-sequence-record manifests",
        )
        return [path for _, path in indexed_manifests]

    data = json.loads(record_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("runs"), list):
        indexed_manifests: list[tuple[int, Path]] = []
        for run in sorted(data["runs"], key=lambda item: int(item.get("rep", 0))):
            manifest = run.get("reusable_manifest") or run.get("sampled_manifest")
            if not manifest:
                continue
            indexed_manifests.append(
                (int(run.get("rep", 0)), _resolve_path(str(manifest), repo_root))
            )
        if indexed_manifests:
            _validate_contiguous_indexes(
                [index for index, _ in indexed_manifests],
                "--task-sequence-record runs",
            )
            return [path for _, path in indexed_manifests]

    raise ValueError(
        "--task-sequence-record must be either a directory containing "
        "repNNN_sampled_manifest.json files or a JSON record with runs[*].reusable_manifest."
    )


def _sequence_pairs(path: Path) -> list[dict[str, object]]:
    """Read unresolved task-pair entries for sequence concatenation."""
    document = json.loads(path.read_text(encoding="utf-8"))
    entries = document.get("pairs") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"{path} is not a task sequence: expected an object with a non-empty "
            '"pairs" list.'
        )
    return entries


def _merged_sequence_dir(records: list[Path]) -> Path:
    """Place merged sequences beside the first record, named for their source directories."""
    head = records[0]
    parts = [head.name if head.is_dir() else head.stem]
    for record in records[1:]:
        name = record.name if record.is_dir() else record.stem
        parts.append(
            name[len(SEQUENCE_DIR_PREFIX) :]
            if name.startswith(SEQUENCE_DIR_PREFIX)
            else name
        )
    return head.parent / "+".join(parts)


def _write_merged_sequences(
    records: list[Path],
    per_record: list[list[Path]],
    repo_root: Path,
) -> list[Path]:
    """Write rep N of every record, concatenated, as one sequence per repeat."""
    if len({len(manifests) for manifests in per_record}) != 1:
        raise ValueError(
            "--task-sequence-record records must cover the same repeats to be "
            "concatenated, because repeat N is read from each of them; got "
            + ", ".join(
                f"{record} with {len(manifests)}"
                for record, manifests in zip(records, per_record)
            )
        )

    # Validate all repeats before writing any merged files.
    merged_pairs: list[list[dict[str, object]]] = []
    with tempfile.TemporaryDirectory() as scratch:
        for index, group in enumerate(zip(*per_record), start=1):
            pairs = [entry for path in group for entry in _sequence_pairs(path)]
            probe = Path(scratch) / f"rep{index:03d}_sampled_manifest.json"
            _write_sequence(probe, pairs)
            try:
                # Reject invalid pairs and tasks repeated across episodes before launch.
                read_task_pairs(probe, repo_root)
            except ValueError as error:
                raise ValueError(
                    "Concatenating "
                    + " + ".join(
                        str(_relative_to_repo(path, repo_root)) for path in group
                    )
                    + f" is not a run: {error}"
                ) from error
            merged_pairs.append(pairs)

    merged_dir = _merged_sequence_dir(records)
    stale = (
        sorted(merged_dir.glob("rep*_sampled_manifest.json"))
        if merged_dir.is_dir()
        else []
    )
    if stale and not (merged_dir / MERGED_SEQUENCE_RECORD).exists():
        raise ValueError(
            f"{merged_dir} already holds task sequences that no merge wrote (there is "
            f"no {MERGED_SEQUENCE_RECORD} beside them), so this merge will not "
            "overwrite them. Move that directory aside or rename a source."
        )
    merged_dir.mkdir(parents=True, exist_ok=True)
    for path in stale[len(merged_pairs) :]:
        # Remove only obsolete repeats; replace retained files atomically for concurrent runs.
        path.unlink()

    merged: list[Path] = []
    for index, pairs in enumerate(merged_pairs, start=1):
        path = merged_dir / f"rep{index:03d}_sampled_manifest.json"
        _write_sequence(path, pairs)
        merged.append(path)

    counts = sorted({len(pairs) for pairs in merged_pairs})
    (merged_dir / MERGED_SEQUENCE_RECORD).write_text(
        json.dumps(
            {
                "description": (
                    "Written by experiments.cli: the run sequences of the source "
                    "records below, concatenated rep by rep, one file per repeat. "
                    "Ordinary sequence files -- ids only, resolved against the pool "
                    "manifests under task/ at load time -- so this directory can "
                    "itself be passed as a single --task-sequence-record."
                ),
                "sources": [
                    str(_relative_to_repo(record, repo_root)) for record in records
                ],
                "reps": len(merged),
                "pairs_per_run": counts[0] if len(counts) == 1 else counts,
                "generated": (
                    "Rewritten by every launch that passes the same records to "
                    "--task-sequence-record, in that order; edits here are lost."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return merged


def _write_sequence(path: Path, pairs: list[dict[str, object]]) -> None:
    """Atomically replace a sequence file so concurrent readers never see partial output."""
    scratch = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    scratch.write_text(
        json.dumps({"pairs": pairs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(scratch, path)


def _relative_to_repo(path: Path, repo_root: Path) -> Path:
    try:
        return path.relative_to(repo_root)
    except ValueError:
        return path


def _validate_contiguous_indexes(indexes: list[int], source: str) -> None:
    expected = list(range(1, len(indexes) + 1))
    if indexes != expected:
        raise ValueError(
            f"{source} must use contiguous 1-based indexes; "
            f"got {indexes[:10]}{'...' if len(indexes) > 10 else ''}."
        )


@dataclass(frozen=True)
class RunRecords:
    task_manifests: list[Path]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeated experiment sequences.")
    task_group = parser.add_argument_group("task selection")
    task_group.add_argument(
        "--task-sequence-record",
        nargs="+",
        default=[],
        metavar="RECORD",
        help=(
            "Reuse task manifests from JSON records or directories containing "
            "repNNN_sampled_manifest.json files. Several records concatenate rep by "
            "rep -- repeat N runs the first record's rep N, then the second's, in the "
            "order given -- and the merge is written beside the first record as a "
            "sequence directory of its own. Defaults to task/task_sequences_50x10."
        ),
    )
    task_group.add_argument("--repeats", type=int, default=1)
    task_group.add_argument(
        "--start-index",
        type=int,
        default=1,
        help=(
            "1-based starting index for run labels and task manifests."
        ),
    )

    add_model_arguments(parser)

    protocol_group = parser.add_argument_group("protocol")
    protocol_group.add_argument("--char-limit", type=int, default=200)
    protocol_group.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
        help=MAX_ROUNDS_HELP,
    )
    protocol_group.add_argument(
        "--verdict-policy-spans",
        default="",
        help="Per-episode verdict-policy spans, e.g. summary-allowed:5,raw-only:*.",
    )
    protocol_group.add_argument(
        "--throttle-spans",
        default="",
        help="Per-episode throttle spans, e.g. no-throttle:3,throttled:*.",
    )

    add_feedback_arguments(parser)

    add_memory_arguments(parser, default_scope="full-history")
    add_reward_scheme_arguments(parser)

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--output-dir",
        default="results",
        help="Directory for run JSON files.",
    )
    output_group.add_argument("--dry-run", action="store_true")
    output_group.add_argument("--quiet", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 1:
        raise ValueError("--start-index must be at least 1.")
    validate_run_args(args)


def _check_record_capacity(
    values: list[object],
    *,
    args: argparse.Namespace,
    description: str,
) -> None:
    last_index = args.start_index + args.repeats - 1
    if values and last_index > len(values):
        raise ValueError(
            f"Requested {description} indexes {args.start_index}-{last_index}, "
            f"but the record contains only {len(values)} entr"
            f"{'y' if len(values) == 1 else 'ies'}."
        )


def _load_run_records(
    args: argparse.Namespace,
    repo_root: Path,
) -> RunRecords:
    task_manifests = _load_task_sequence_manifests(
        args.task_sequence_record,
        repo_root,
    )
    _check_record_capacity(task_manifests, args=args, description="task sequence")
    return RunRecords(task_manifests=task_manifests)


def _verdict_policy_label(args: argparse.Namespace) -> str:
    if args.verdict_policy_spans:
        return "verdict-policy-spans"
    return DEFAULT_VERDICT_POLICY


def _run_label(args: argparse.Namespace, run_index: int) -> str:
    label = _setting_label(run_index, _verdict_policy_label(args), args.max_rounds)
    suffixes: list[str] = []
    if args.throttle_spans:
        suffixes.append("throttle-spans")
    suffixes.append(f"memory-{args.cross_episode_memory_scope}")
    for agent_id in AGENT_IDS:
        max_episodes = getattr(args, f"{agent_id}_cross_episode_memory_length")
        if max_episodes >= 0:
            suffixes.append(
                f"{agent_id.replace('_', '-')}-memory-episodes-{max_episodes}"
            )
    # Include the reward scheme, including the default.
    suffixes.append(reward_scheme_from_args(args).label)
    optional_suffixes = (
        ("reward", args.reward),
        ("verdict-review", args.verdict_review),
        ("reflection", args.reflection),
    )
    suffixes.extend(name for name, enabled in optional_suffixes if enabled)
    return "_".join([label, *suffixes])


def _build_runner_command(
    *,
    args: argparse.Namespace,
    records: RunRecords,
    run_index: int,
) -> list[str]:
    sequence_index = run_index - 1
    label = _run_label(args, run_index)
    command = [
        sys.executable,
        "-m",
        "experiments.runner",
        "--cross-episode-memory-scope",
        args.cross_episode_memory_scope,
        "--verdict-policy",
        DEFAULT_VERDICT_POLICY,
        "--char-limit",
        str(args.char_limit),
        "--max-rounds",
        str(args.max_rounds),
        "--manifest",
        str(records.task_manifests[sequence_index]),
        "--run-label",
        label,
        "--output-dir",
        args.output_dir,
    ]
    if args.bob_model == "controlled":
        if args.controlled_bob_observed_verdict:
            command.append("--controlled-bob-observed-verdict")
        for name in (
            "controlled_bob_verdict",
            "controlled_bob_messages",
            "controlled_bob_cache",
        ):
            value = getattr(args, name)
            if name == "controlled_bob_cache":
                value = str(Path(value) / f"rep{sequence_index + 1:03d}")
            if value:
                command.extend(["--" + name.replace("_", "-"), value])
    for agent_id in AGENT_IDS:
        flag = agent_id.replace("_", "-")
        command.extend([f"--{flag}-model", getattr(args, f"{agent_id}_model")])
        if not (agent_id == "bob" and replays_controlled_bob(args)):
            command.extend(
                [
                    f"--{flag}-reasoning-effort",
                    getattr(args, f"{agent_id}_reasoning_effort"),
                    f"--{flag}-temperature",
                    str(getattr(args, f"{agent_id}_temperature")),
                    f"--{flag}-max-output-tokens",
                    str(getattr(args, f"{agent_id}_max_output_tokens")),
                ]
            )
        command.extend(
            [
                f"--{flag}-cross-episode-memory-length",
                str(getattr(args, f"{agent_id}_cross_episode_memory_length")),
            ]
        )
    for name in ("verdict-policy-spans", "throttle-spans"):
        value = getattr(args, name.replace("-", "_"))
        if value:
            command.extend([f"--{name}", value])
    if args.reward:
        command.extend(["--reward-scope", args.reward_scope, "--reward-type", args.reward_type])
    for name in ("reflection", "reward", "verdict-review"):
        enabled = getattr(args, name.replace("-", "_"))
        command.append(f"--{name}" if enabled else f"--no-{name}")
    outcome_flags = (
        ("--quiet", args.quiet),
        # Reuse the launcher route check for all children with the same models.
        ("--no-preflight", True),
    )
    command.extend(flag for flag, enabled in outcome_flags if enabled)
    return command


def main() -> None:
    repo_root = Path.cwd().resolve()
    load_dotenv(repo_root / ".env")
    args = _build_parser().parse_args()
    # Dry runs must not make route-check requests.
    args.no_preflight = args.no_preflight or args.dry_run
    _validate_args(args)
    records = _load_run_records(args, repo_root)

    for run_index in range(args.start_index, args.start_index + args.repeats):
        command = _build_runner_command(
            args=args,
            records=records,
            run_index=run_index,
        )
        print("$ " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
