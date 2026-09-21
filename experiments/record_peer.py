"""Record the controlled peer's trajectories: Bob alone, with no Alice and no interaction.

Each episode runs Bob's task phase and stores both message sets it can later transmit:
the summaries it writes about its own work, and the leading characters of its raw log.
Replaying the cache with ``--bob-model controlled`` selects one of them.

    python -m experiments.record_peer \
        --model gemini/gemini-3.1-flash-lite --reasoning-effort high \
        --task-sequence-record task/task_sequences_50x10 --repeats 50 \
        --cache results/peer-cache
"""

import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv

from experiments.cli import _sequence_record_manifests
from experiments.controlled import cache_path, save_bob
from experiments.episode_runner import (
    _open_episode,
    _prepare_pair,
    _run_agent_attempts,
    log_progress,
)
from experiments.llm import REASONING_EFFORT_CHOICES
from experiments.models import AgentConfig, AgentRuntime, EpisodeRunConfig
from experiments.prompts.system import initial_agent_messages
from experiments.protocol.state import agent_state, set_phase
from experiments.tasks import DEFAULT_TASK_SEQUENCES, read_task_pairs
from experiments.usage import LLM_USAGE_JOURNAL_FILENAME


def record_episode(*, pair, agent_config, repo_root, config, label):
    """Run Bob's task phase for one episode and write its cache entry.

    Returns None when the episode is already cached, so a resumed run neither
    repeats nor overwrites it.
    """
    prepared = _prepare_pair(pair=pair, repo_root=repo_root, config=config)
    if cache_path(prepared, config).exists():
        return None
    log_progress(f"Recording {label}", config.verbose)
    runtime = AgentRuntime(
        agent_id="bob",
        config=agent_config,
        messages=initial_agent_messages(agent_id="bob", max_rounds=config.max_rounds),
    )
    _open_episode(runtime.messages, prepared.agents["bob"].own_task_prompt)
    set_phase(prepared.state, "task")
    _run_agent_attempts(
        runtime=runtime,
        state=prepared.state,
        phase="task",
        tool_choice="required",
        episode_id=str(prepared.state["episode_id"]),
        round_index=None,
        reasoning_traces=[],
        usage_records=[],
        config=config,
    )
    path = save_bob(prepared, runtime, config, [], [])
    shutil.rmtree(agent_state(prepared.state, "bob")["task_tmp_dir"], ignore_errors=True)
    return path


def main() -> None:
    repo_root = Path.cwd().resolve()
    load_dotenv(repo_root / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="The peer's LiteLLM model.")
    parser.add_argument(
        "--reasoning-effort", required=True, choices=REASONING_EFFORT_CHOICES
    )
    parser.add_argument(
        "--task-sequence-record",
        default=DEFAULT_TASK_SEQUENCES,
        help=f"Task sequences to record. Defaults to {DEFAULT_TASK_SEQUENCES}.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument(
        "--cache", required=True, help="Directory for the per-repeat caches."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    record = Path(args.task_sequence_record)
    manifests = _sequence_record_manifests(
        record if record.is_absolute() else repo_root / record, repo_root
    )
    agent_config = AgentConfig(
        model=args.model, reasoning_effort=args.reasoning_effort
    )
    cache_root = Path(args.cache)
    if not cache_root.is_absolute():
        cache_root = repo_root / cache_root

    for rep in range(args.start_index, args.start_index + args.repeats):
        pairs = read_task_pairs(manifests[rep - 1], repo_root)
        for episode_index, pair in enumerate(pairs):
            config = EpisodeRunConfig(
                controlled_bob_cache=str(cache_root / f"rep{rep:03d}"),
                episode_index=episode_index,
                temp_run_id=f"record_rep{rep:03d}",
                verbose=not args.quiet,
                usage_journal_path=cache_root / LLM_USAGE_JOURNAL_FILENAME,
                usage_invocation_id=f"record_rep{rep:03d}_ep{episode_index:04d}",
            )
            record_episode(
                pair=pair,
                agent_config=agent_config,
                repo_root=repo_root,
                config=config,
                label=f"rep{rep:03d} episode {episode_index + 1}/{len(pairs)}",
            )
    print(f"cache written to {cache_root}")


if __name__ == "__main__":
    main()
