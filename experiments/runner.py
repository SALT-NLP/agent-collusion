"""Run-level task selection, persistence, and command-line entry point."""

import argparse
import json
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from experiments.controlled import MESSAGE_POLICIES, cached_model_settings
from experiments.episode_runner import log_progress, run_one_episode
from experiments.episode_schedules import (
    VERDICT_POLICIES,
    expand_verdict_policy_spans,
    expand_throttle_spans,
)
from experiments.llm import REASONING_EFFORT_CHOICES, validate_model_route
from experiments.memory.cross_episode import (
    CROSS_EPISODE_MEMORY_SCOPES,
    set_cross_episode_memory_from_results,
)
from experiments.memory.messages import restore_episode_boundaries
from experiments.models import (
    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
    AgentConfig,
    EpisodeRunConfig,
)
from experiments.output import build_run_output, write_run_output
from experiments.prompts.system import initial_agent_messages
from experiments.protocol import (
    DEFAULT_MAX_ROUNDS,
    MEMORY_SCOPE,
)
from experiments.protocol.rewards import (
    DEFAULT_REWARD_SHARING,
    DEFAULT_REWARD_TERMS,
    REWARD_SHARINGS,
    REWARD_TERMS,
    RewardScheme,
    reward_scheme_from_args,
)
from experiments.protocol.state import AGENT_IDS
from experiments.tasks import (
    DEFAULT_TASK_SEQUENCES,
    display_task_id,
    read_task_pairs,
    sanitize_label,
)
from experiments.usage import LLM_USAGE_JOURNAL_FILENAME

MAX_ROUNDS_HELP = (
    "Communication rounds per episode, exactly: the phase runs all of them with no "
    "early exit. This is rounds within an episode, not episodes; the task sequence "
    "or sampled task pairs determine the episode count. Every episode of a run gets "
    f"the same count, because the agents are told it. Default {DEFAULT_MAX_ROUNDS}."
)

@dataclass
class SelectedPairs:
    """The run's episodes, each one a pair of manifest tasks."""

    pairs: list[tuple[dict[str, Any], dict[str, Any]]]
    manifest_path: Path


def _agent_flag(agent_id: str, suffix: str) -> str:
    return "--" + agent_id.replace("_", "-") + "-" + suffix


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Configure each agent's model and completion settings, including route pre-flight."""
    group = parser.add_argument_group("models")
    group.add_argument(
        "--controlled-bob-verdict", choices=("accept", "reject"), default="reject"
    )
    group.add_argument(
        "--controlled-bob-messages",
        choices=MESSAGE_POLICIES,
        default=MESSAGE_POLICIES[0],
        help="Which recorded messages the controlled peer transmits.",
    )
    group.add_argument(
        "--controlled-bob-observed-verdict", action="store_true",
        help="Append Bob's fixed verdict to Alice's feedback.",
    )
    group.add_argument(
        "--controlled-bob-cache",
        default="",
        help="Directory holding immutable per-episode Bob trajectories.",
    )
    group.add_argument(
        "--no-preflight",
        action="store_true",
        help=(
            "Skip the one-request check that each agent's model can be called. "
            "The check needs network; without it an unreachable route is found "
            "by the first episode instead, after the run directory exists."
        ),
    )
    for agent_id in AGENT_IDS:
        group.add_argument(
            _agent_flag(agent_id, "model"),
            required=True,
            help=f"{agent_id} LiteLLM model.",
        )
        group.add_argument(
            _agent_flag(agent_id, "reasoning-effort"),
            choices=REASONING_EFFORT_CHOICES,
            default="",
            help=(
                f"{agent_id} reasoning effort; default lets the provider choose. "
                "Required unless the cache supplies it."
            ),
        )
        group.add_argument(
            _agent_flag(agent_id, "temperature"),
            type=float,
            default=None,
            help=f"{agent_id} sampling temperature. Default {DEFAULT_LLM_TEMPERATURE}.",
        )
        group.add_argument(
            _agent_flag(agent_id, "max-output-tokens"),
            type=int,
            default=None,
            help=(
                f"{agent_id} maximum completion tokens, including reasoning tokens "
                "where the provider counts them. Default "
                f"{DEFAULT_LLM_MAX_OUTPUT_TOKENS}."
            ),
        )


def add_memory_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_scope: str,
) -> None:
    group = parser.add_argument_group("cross-episode memory")
    group.add_argument(
        "--cross-episode-memory-scope",
        choices=CROSS_EPISODE_MEMORY_SCOPES,
        default=default_scope,
    )
    for agent_id in AGENT_IDS:
        group.add_argument(
            _agent_flag(agent_id, "cross-episode-memory-length"),
            type=int,
            default=-1,
            help=f"-1 keeps all {agent_id} history; 0 keeps none.",
        )


REWARD_SCHEME_HELP = {
    "acceptance": (
        "paid for being accepted, whatever the work was worth: ground truth does not "
        "enter the reward at all"
    ),
    "verdict-accuracy": (
        "paid whenever the verdict matches the truth, so a deserved reject pays what a "
        "deserved accept does"
    ),
}


def add_reward_scheme_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("reward scheme")
    group.add_argument(
        "--reward-scope",
        choices=REWARD_SHARINGS,
        default=None,
        help="Reward scope: shared (default) or separate. Requires --reward.",
    )
    group.add_argument(
        "--reward-type",
        choices=REWARD_TERMS,
        default=None,
        help=(
            "Reward type: verdict-accuracy (default) or acceptance. Requires --reward. "
            + "; ".join(f"{name}: {text}" for name, text in REWARD_SCHEME_HELP.items())
        ),
    )


def add_verdict_review_arguments(group) -> None:
    group.add_argument(
        "--verdict-review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Review verdict correctness (enabled by default): shared rewards review "
            "both verdicts; separate rewards review the peer's verdict on your task. "
            "Use --no-verdict-review to disable. Independent of reward feedback."
        ),
    )


def add_feedback_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("feedback and reflection")
    group.add_argument(
        "--reward", action=argparse.BooleanOptionalAction, default=True,
        help="Include the reward objective and reward feedback (default: enabled).",
    )
    group.add_argument(
        "--reflection", action=argparse.BooleanOptionalAction, default=None,
        help="Private reflection after each episode; defaults to enabled with --reward.",
    )
    add_verdict_review_arguments(group)


def _resolve_feedback_settings(args: argparse.Namespace) -> None:
    """Resolve reward defaults and validate dependent feedback settings."""
    if not args.reward and (args.reward_scope is not None or args.reward_type is not None):
        raise ValueError("--reward-scope and --reward-type require --reward.")
    if args.reward:
        if args.reward_scope is None:
            args.reward_scope = DEFAULT_REWARD_SHARING
        if args.reward_type is None:
            args.reward_type = DEFAULT_REWARD_TERMS
    if args.reflection is None:
        args.reflection = args.reward
    if args.reflection and not args.reward:
        raise ValueError("--reflection requires --reward.")


def build_run_parser() -> argparse.ArgumentParser:
    """Create the internal single-run parser."""
    parser = argparse.ArgumentParser(
        description="Run one collusion-safety two-agent experiment."
    )
    task_group = parser.add_argument_group("task selection")
    task_group.add_argument(
        "--manifest",
        default="",
        help=(
            "Stored task sequence: pairs of task ids, resolved against the pool "
            "manifests and run in the order written. Defaults to the first "
            f"sequence in {DEFAULT_TASK_SEQUENCES}."
        ),
    )

    add_model_arguments(parser)

    protocol_group = parser.add_argument_group("protocol")
    protocol_group.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
        help=MAX_ROUNDS_HELP,
    )
    protocol_group.add_argument("--char-limit", type=int, default=200)
    protocol_group.add_argument(
        "--throttle-spans",
        default="",
        help="Per-episode spans, e.g. no-throttle:2,throttled:*.",
    )
    protocol_group.add_argument(
        "--verdict-policy",
        choices=VERDICT_POLICIES,
        default="raw-only",
    )
    protocol_group.add_argument(
        "--verdict-policy-spans",
        default="",
        help="Per-episode spans, e.g. raw-only:2,summary-allowed:*.",
    )

    add_memory_arguments(parser, default_scope="full-history")

    add_feedback_arguments(parser)

    add_reward_scheme_arguments(parser)

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--run-label", default="")
    output_group.add_argument("--output-dir", default="results")
    output_group.add_argument("--quiet", action="store_true")
    output_group.add_argument(
        "--resume-from",
        default="",
        help=(
            "Continue a run whose run.json is at this path, starting at its first "
            "unfinished episode. The manifest and every setting an episode's "
            "behaviour depends on must match the record, and only "
            "--cross-episode-memory-scope full-history with no episode "
            "cap can be resumed -- see _resume_state."
        ),
    )
    return parser


def _validate_rounds(args: argparse.Namespace) -> None:
    """Validate the communication budget and message limit."""
    if args.max_rounds < 1 and not (
        getattr(args, "bob_model", "") == "controlled" and args.max_rounds == 0
    ):
        raise ValueError("--max-rounds must be at least 1.")
    if args.char_limit < 1:
        raise ValueError("--char-limit must be at least 1.")


def replays_controlled_bob(args: argparse.Namespace) -> bool:
    """Whether Bob is a cached trajectory replayed from disk, so it calls no model."""
    return getattr(args, "bob_model", "") == "controlled"


def _resolve_model_settings(args: argparse.Namespace) -> None:
    """Apply completion defaults; a replayed Bob takes its settings from the cache."""
    for agent_id in AGENT_IDS:
        settings = {
            "reasoning-effort": getattr(args, f"{agent_id}_reasoning_effort"),
            "temperature": getattr(args, f"{agent_id}_temperature"),
            "max-output-tokens": getattr(args, f"{agent_id}_max_output_tokens"),
        }
        if agent_id == "bob" and replays_controlled_bob(args):
            supplied = [name for name, value in settings.items() if value not in (None, "")]
            if supplied:
                raise ValueError(
                    "A controlled Bob takes its model settings from the cache; "
                    "drop " + ", ".join(f"--bob-{name}" for name in supplied)
                )
            continue
        if not settings["reasoning-effort"]:
            raise ValueError(f"{_agent_flag(agent_id, 'reasoning-effort')} is required.")
        if settings["temperature"] is None:
            setattr(args, f"{agent_id}_temperature", DEFAULT_LLM_TEMPERATURE)
        if settings["max-output-tokens"] is None:
            setattr(args, f"{agent_id}_max_output_tokens", DEFAULT_LLM_MAX_OUTPUT_TOKENS)


def validate_run_args(args: argparse.Namespace) -> None:
    """Reject internally inconsistent single-run options."""
    # Default to the paper's fixed ten-episode sequences.
    if hasattr(args, "task_sequence_record"):
        if not args.task_sequence_record:
            args.task_sequence_record = [DEFAULT_TASK_SEQUENCES]
    elif not args.manifest:
        args.manifest = f"{DEFAULT_TASK_SEQUENCES}/rep001_sampled_manifest.json"
    _validate_rounds(args)
    _resolve_model_settings(args)
    for agent_id in AGENT_IDS:
        if getattr(args, f"{agent_id}_cross_episode_memory_length") < -1:
            raise ValueError(
                f"{_agent_flag(agent_id, 'cross-episode-memory-length')} "
                "must be -1 or greater."
            )
        temperature = getattr(args, f"{agent_id}_temperature")
        if temperature is not None and (not math.isfinite(temperature) or temperature < 0):
            raise ValueError(
                f"{_agent_flag(agent_id, 'temperature')} must be a finite, "
                "non-negative number."
            )
        max_output_tokens = getattr(args, f"{agent_id}_max_output_tokens")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError(
                f"{_agent_flag(agent_id, 'max-output-tokens')} must be at least 1."
            )
    _resolve_feedback_settings(args)
    # Validate manually constructed options as well as parser-generated values.
    reward_scheme_from_args(args)
    controlled = getattr(args, "bob_model", "") == "controlled"
    if controlled:
        if not args.controlled_bob_cache:
            raise ValueError("--bob-model controlled requires --controlled-bob-cache")
        # Filled in from the cache, which names the model that recorded it.
        args.controlled_bob_task_model = ""
        if args.max_rounds not in (0, 5) or args.char_limit < 200:
            raise ValueError(
                "Controlled Bob requires --max-rounds 0 or 5 and --char-limit >= 200"
            )
    elif args.controlled_bob_cache:
        raise ValueError("Controlled Bob options require --bob-model controlled")
    for agent_id in AGENT_IDS:
        if agent_id == "bob" and replays_controlled_bob(args):
            continue
        validate_model_route(
            getattr(args, f"{agent_id}_model"),
            reasoning_effort=getattr(args, f"{agent_id}_reasoning_effort"),
            temperature=getattr(args, f"{agent_id}_temperature"),
            max_output_tokens=getattr(args, f"{agent_id}_max_output_tokens"),
            probe=not args.no_preflight,
        )


def _create_output_path(args: argparse.Namespace, repo_root: Path) -> Path:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    label = sanitize_label(args.run_label)
    suffix = f"{random.SystemRandom().randrange(16**6):06x}"
    directory_name = (
        f"run_{label}_{timestamp}_{suffix}" if label else f"run_{timestamp}_{suffix}"
    )
    run_directory = output_dir / directory_name
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory / "run.json"


def _load_selected_tasks(
    *,
    args: argparse.Namespace,
    repo_root: Path,
) -> SelectedPairs:
    """Resolve the stored episode sequence against the task pool."""
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    pairs = read_task_pairs(manifest_path, repo_root)
    if not pairs:
        raise ValueError(f"{manifest_path} selected no episodes.")
    return SelectedPairs(pairs, manifest_path)


def _episode_policy_sequences(
    args: argparse.Namespace,
    episode_count: int,
) -> tuple[list[str], list[str]]:
    """Resolve one verdict and throttle policy per episode, reporting trimmed entries."""

    def notice(message: str) -> None:
        log_progress(f"  [schedule: {message}]", True)

    verdict = expand_verdict_policy_spans(
        verdict_policy_spans=args.verdict_policy_spans,
        episode_count=episode_count,
        default_policy=args.verdict_policy,
        notice=notice,
    )
    throttle = expand_throttle_spans(
        throttle_spans=args.throttle_spans,
        episode_count=episode_count,
        notice=notice,
    )
    return verdict, throttle


def _resume_state(
    resume_from: str,
    *,
    args: argparse.Namespace,
    base_agent_messages: dict[str, list[dict[str, Any]]],
    reward_scheme: RewardScheme,
    manifest_path: Path | None,
    episode_count: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Restore completed episodes and agent histories from a compatible run record.

    Rebuild episode boundaries and reject configuration or system-prompt mismatches.
    """
    record = json.loads(Path(resume_from).read_text())
    results = record.get("results") or []
    config = record.get("run_config") or {}
    if not results:
        raise ValueError(f"{resume_from} has no finished episodes to resume from")
    if len(results) >= episode_count:
        raise ValueError(
            f"{resume_from} already has {len(results)} of {episode_count} episodes, "
            "so there is nothing left to run"
        )

    # Stripped snapshots cannot restore within-episode markers or capped episode counts.
    if args.cross_episode_memory_scope != "full-history":
        raise ValueError(
            "--resume-from only supports --cross-episode-memory-scope "
            "full-history, because a stripped record has no "
            "outcome-feedback delimiter for the other scopes to cut at; "
            f"got {args.cross_episode_memory_scope}"
        )
    capped = [
        agent_id
        for agent_id in AGENT_IDS
        if getattr(args, f"{agent_id}_cross_episode_memory_length") != -1
    ]
    if capped:
        raise ValueError(
            "--resume-from needs every "
            "--<agent>-cross-episode-memory-length to be -1; capped: "
            + ", ".join(capped)
        )

    expected = {
        "cross_episode_memory_scope": args.cross_episode_memory_scope,
        "max_rounds": args.max_rounds,
        "char_limit": args.char_limit,
        "verdict_evaluation": reward_scheme.verdict_review_direction(args.verdict_review),
        "reflection": args.reflection,
        "reward": args.reward,
        "reward_scheme": f"{reward_scheme.sharing}-{reward_scheme.terms}",
    }
    for key in (
        "controlled_bob_task_model",
        "controlled_bob_verdict",
        "controlled_bob_cache",
        "controlled_bob_messages",
        "controlled_bob_observed_verdict",
    ):
        if (
            args.bob_model == "controlled"
            or config.get("models", {}).get("bob") == "controlled"
        ):
            expected[key] = getattr(args, key)
    if (
        config.get("models", {}).get("bob") == "controlled"
        and args.bob_model != "controlled"
    ):
        raise ValueError("Cannot resume controlled Bob as a live peer")
    if args.bob_model == "controlled":
        config.setdefault("controlled_bob_observed_verdict", False)
    mismatched = [
        f"{key}: record has {config[key]!r}, this run has {value!r}"
        for key, value in expected.items()
        if key in config and config[key] != value
    ]
    if manifest_path is not None and config.get("manifest"):
        if str(Path(config["manifest"]).resolve()) != str(manifest_path.resolve()):
            mismatched.append(
                f"manifest: record has {config['manifest']!r}, "
                f"this run has {str(manifest_path)!r}"
            )
    # Require the current system prompt to match the restored opening turn.
    recorded_prompts = (config.get("prompt_texts") or {}).get("system_prompts") or {}
    for agent_id in AGENT_IDS:
        recorded = recorded_prompts.get(agent_id)
        mine = base_agent_messages[agent_id][0].get("content")
        if recorded is not None and recorded != mine:
            mismatched.append(f"{agent_id} system prompt differs from the record")
    if mismatched:
        raise ValueError(
            f"Cannot resume from {resume_from}, it was written under a different "
            "configuration:\n  " + "\n  ".join(mismatched)
        )

    agent_messages: dict[str, list[dict[str, Any]]] = {}
    for agent_id in AGENT_IDS:
        snapshots: list[tuple[str, list[dict[str, Any]]]] = []
        for episode in results:
            agents = episode.get("agents") or {}
            if agent_id not in agents:
                raise ValueError(
                    f"{resume_from} episode "
                    f"{episode.get('episode_id') or '(unknown)'} has no {agent_id} "
                    "record, so its history cannot be restored"
                )
            snapshots.append(
                (
                    str(episode.get("episode_id") or ""),
                    agents[agent_id].get("messages") or [],
                )
            )
        agent_messages[agent_id] = restore_episode_boundaries(
            base_agent_messages[agent_id],
            snapshots,
        )
    return deepcopy(results), agent_messages


def _run_selected_episodes(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    output_path: Path,
    selected: SelectedPairs,
    verdict_policy_by_episode: list[str],
    throttle_by_episode: list[str],
) -> None:
    results: list[dict[str, Any]] = []
    if replays_controlled_bob(args):
        # Each cache names the settings that wrote it, so replaying asks for no model.
        recorded = cached_model_settings(args.controlled_bob_cache)
        args.controlled_bob_task_model = recorded["model"]
        args.bob_reasoning_effort = recorded["reasoning_effort"]
        args.bob_temperature = recorded["temperature"]
        args.bob_max_output_tokens = recorded["max_output_tokens"]
    # Resolve one reward scheme for both scoring and prompt construction.
    reward_scheme = reward_scheme_from_args(args)
    # Build one system prompt per agent; enable the reward objective with reward feedback.
    base_agent_messages = {
        agent_id: initial_agent_messages(
            agent_id=agent_id,
            max_rounds=5
            if agent_id == "bob" and args.bob_model == "controlled"
            else args.max_rounds,
            reward_objective=args.reward
            and not (agent_id == "bob" and args.bob_model == "controlled"),
            reward_scheme=reward_scheme,
        )
        for agent_id in AGENT_IDS
    }
    agent_messages = {
        agent_id: deepcopy(messages)
        for agent_id, messages in base_agent_messages.items()
    }
    # Include completed episodes in every snapshot when resuming.
    completed_episodes = 0
    if args.resume_from:
        results, agent_messages = _resume_state(
            args.resume_from,
            args=args,
            base_agent_messages=base_agent_messages,
            reward_scheme=reward_scheme,
            manifest_path=selected.manifest_path,
            episode_count=len(selected.pairs),
        )
        completed_episodes = len(results)

    def persist() -> None:
        write_run_output(
            output_path,
            build_run_output(
                args=args,
                manifest_path=selected.manifest_path,
                pair_count=len(selected.pairs),
                initial_verdict_policy=args.verdict_policy,
                verdict_policy_by_episode=verdict_policy_by_episode,
                throttle_policy_by_episode=throttle_by_episode,
                results=results,
            ),
        )

    persist()
    log_progress(f"Saving incremental results to: {output_path}", True)
    temporary_run_id = sanitize_label(output_path.parent.name)
    usage_journal_path = output_path.parent / LLM_USAGE_JOURNAL_FILENAME
    usage_invocation_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_"
        f"{random.SystemRandom().randrange(16**6):06x}"
    )
    verbose = not args.quiet
    agent_configs = {
        agent_id: AgentConfig(
            model=(
                args.controlled_bob_task_model
                if agent_id == "bob" and args.bob_model == "controlled"
                else getattr(args, f"{agent_id}_model")
            ),
            reasoning_effort=getattr(args, f"{agent_id}_reasoning_effort"),
            temperature=getattr(args, f"{agent_id}_temperature"),
            max_output_tokens=getattr(args, f"{agent_id}_max_output_tokens"),
        )
        for agent_id in AGENT_IDS
    }
    max_memory_episodes_by_agent = {
        agent_id: getattr(args, f"{agent_id}_cross_episode_memory_length")
        for agent_id in AGENT_IDS
    }
    # Report the reward scheme even when it is the default.
    log_progress(
        f"Reward scheme: {reward_scheme.sharing} / {reward_scheme.terms}",
        True,
    )
    if completed_episodes:
        log_progress(
            f"Resuming after {completed_episodes}/{len(selected.pairs)} episodes "
            f"from: {args.resume_from}",
            True,
        )

    for episode_index, pair in enumerate(selected.pairs):
        if episode_index < completed_episodes:
            continue
        throttled = throttle_by_episode[episode_index] == "throttled"
        verdict_policy = verdict_policy_by_episode[episode_index]
        group = throttle_by_episode[episode_index]
        shown = " + ".join(display_task_id(task) for task in pair)
        internal = " + ".join(str(task["task_id"]) for task in pair)
        log_progress(
            f"Running episode {episode_index + 1}/{len(selected.pairs)}: {shown} "
            f"({internal}) [{group}, "
            f"memory={MEMORY_SCOPE}/{args.cross_episode_memory_scope}"
            f"/max_episodes={max_memory_episodes_by_agent}, "
            f"verdict-policy={verdict_policy}]",
            True,
        )
        result = run_one_episode(
            pair=pair,
            agent_configs=agent_configs,
            repo_root=repo_root,
            config=EpisodeRunConfig(
                throttled=throttled,
                char_limit=args.char_limit,
                max_rounds=args.max_rounds,
                cross_episode_memory_scope=args.cross_episode_memory_scope,
                verdict_policy=verdict_policy,
                reflection=args.reflection,
                reward=args.reward,
                verdict_review=args.verdict_review,
                reward_scheme=reward_scheme,
                controlled_bob_verdict=args.controlled_bob_verdict
                if args.bob_model == "controlled"
                else "",
                controlled_bob_cache=args.controlled_bob_cache,
                controlled_bob_messages=args.controlled_bob_messages,
                controlled_bob_observed_verdict=args.controlled_bob_observed_verdict,
                temp_run_id=temporary_run_id,
                episode_index=episode_index,
                verbose=verbose,
                usage_journal_path=usage_journal_path,
                usage_invocation_id=usage_invocation_id,
            ),
            agent_messages=agent_messages,
        )
        result.update({"group": group, "episode_index": episode_index})
        results.append(result)
        set_cross_episode_memory_from_results(
            agent_messages,
            scope=args.cross_episode_memory_scope,
            episode_id=str(result["episode_id"]),
            max_memory_episodes_by_agent=max_memory_episodes_by_agent,
        )
        persist()
        log_progress(
            "  Saved incremental results after "
            f"{episode_index + 1}/{len(selected.pairs)}",
            True,
        )
    log_progress(f"Saved results to: {output_path}", True)


def main() -> None:
    repo_root = Path.cwd().resolve()
    load_dotenv(repo_root / ".env")
    args = build_run_parser().parse_args()
    validate_run_args(args)
    output_path = _create_output_path(args, repo_root)
    selected = _load_selected_tasks(args=args, repo_root=repo_root)
    verdict_policy_by_episode, throttle_by_episode = _episode_policy_sequences(
        args,
        len(selected.pairs),
    )
    _run_selected_episodes(
        args=args,
        repo_root=repo_root,
        output_path=output_path,
        selected=selected,
        verdict_policy_by_episode=verdict_policy_by_episode,
        throttle_by_episode=throttle_by_episode,
    )


if __name__ == "__main__":
    main()
