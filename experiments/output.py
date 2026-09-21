"""Build and atomically persist run output using internal manifest task IDs."""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from experiments.llm import get_endpoint_metadata, get_litellm_version
from experiments.memory.messages import snapshot_visible_messages
from experiments.models import (
    TaskEvaluation,
    EpisodeOutcomes,
    EpisodeRunConfig,
    PreparedPair,
)
from experiments.prompts.policies import (
    verdict_policy_section,
    reward_objective_text,
)
from experiments.prompts.system import initial_agent_messages
from experiments.protocol import (
    COMMUNICATION_ATTEMPTS,
    MEMORY_SCOPE,
    TASK_ATTEMPTS,
    VERDICT_ATTEMPTS,
)
from experiments.protocol.rewards import reward_scheme_from_args
from experiments.protocol.state import AGENT_IDS, agent_state, display_name, peer_id
from experiments.usage import (
    LLM_USAGE_JOURNAL_FILENAME,
    summarize_llm_usage,
)


def _build_channel_transcript(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return delivered messages in send order, labelled by round and sender."""
    transcript = []
    for event in events:
        if event.get("tool") != "send_message" or not event.get("success"):
            continue
        sender = str(event.get("actor", ""))
        message_type = str(event.get("message_type", ""))
        content = str(event.get("content", ""))
        transcript.append(
            {
                "event_id": event.get("event_id"),
                "round": event.get("round"),
                "sender": sender,
                "receiver": str(event.get("recipient") or peer_id(sender)),
                **({"message_type": message_type} if message_type else {}),
                "content": content,
            }
        )
    return transcript


def _build_agent_result(
    *,
    agent_id: str,
    prepared: PreparedPair,
    evaluations: dict[str, TaskEvaluation],
    outcomes: EpisodeOutcomes,
    repo_root: Path,
    agent_messages: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Serialize one agent's result.

    Ground truth describes its own task; verdict and verdict_correct describe its peer.
    """
    agent = prepared.agents[agent_id]
    task = agent.task
    slot = agent_state(prepared.state, agent_id)
    evaluation = evaluations[agent_id]
    outcome = outcomes.agents[agent_id]
    return {
        "agent_id": agent_id,
        "peer_id": peer_id(agent_id),
        "task_id": task["task_id"],
        "display_id": agent.display_id,
        "task_family": task.get("task_family"),
        "expected_verdict": evaluation.expected_verdict,
        "manifest_expected_verdict": task.get("expected_verdict"),
        "target_expected_verdict": evaluation.target_expected_verdict,
        "spec_key": task.get("spec_key"),
        "code_path": (
            str(agent.code_path.relative_to(repo_root)) if agent.code_path else None
        ),
        "source_path": (
            str(agent.source_path.relative_to(repo_root)) if agent.source_path else None
        ),
        "database_path": (
            str(agent.database_path.relative_to(repo_root))
            if agent.database_path
            else None
        ),
        "function_name": agent.function_name,
        "reference_item_count": task.get("reference_item_count"),
        "code_assessment": deepcopy(slot.get("code_assessment")),
        "code_eval": evaluation.code,
        "extraction_artifact": deepcopy(slot.get("extraction_artifact")),
        "extraction_eval": evaluation.extraction,
        "data_search_artifact": deepcopy(slot.get("data_search_artifact")),
        "data_search_eval": evaluation.data_search,
        "private_raw_log": deepcopy(slot.get("private_raw_log", [])),
        "written_test_files": list(slot.get("written_test_files", [])),
        # Score this agent's verdict against its peer's ground truth.
        "verdict_submission": deepcopy(slot.get("verdict_submission")),
        "verdict": outcome.verdict,
        "verdict_correct": outcome.verdict_correct,
        "verdict_forced": outcome.verdict_forced,
        # Store the reward paid to this agent.
        "reward": outcome.reward,
        "outcome_feedback": outcome.feedback,
        "reflection": outcome.reflection,
        "reflection_prompt": outcome.reflection_prompt,
        "own_task_prompt": agent.own_task_prompt,
        "judging_prompt": agent.judging_prompt,
        "verdict_prompt": agent.verdict_prompt,
        "messages": snapshot_visible_messages(agent_messages[agent_id]),
    }


def build_episode_result(
    *,
    prepared: PreparedPair,
    evaluations: dict[str, TaskEvaluation],
    outcomes: EpisodeOutcomes,
    repo_root: Path,
    config: EpisodeRunConfig,
    reasoning_traces: list[dict[str, Any]],
    usage_records: list[dict[str, Any]],
    agent_messages: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    state = prepared.state
    return {
        "episode_id": state["episode_id"],
        "task_type": prepared.task_type,
        "task_ids": [prepared.agents[a].task["task_id"] for a in AGENT_IDS],
        "display_ids": [prepared.agents[a].display_id for a in AGENT_IDS],
        "memory_scope": MEMORY_SCOPE,
        "verdict_policy": config.verdict_policy,
        "throttled": config.throttled,
        "max_rounds": config.max_rounds,
        # Keep per-agent rewards; separate schemes need not pay equal amounts.
        "reward_by_agent": outcomes.rewards,
        "reward_scheme": config.reward_scheme.label,
        "agents": {
            agent_id: _build_agent_result(
                agent_id=agent_id,
                prepared=prepared,
                evaluations=evaluations,
                outcomes=outcomes,
                repo_root=repo_root,
                agent_messages=agent_messages,
            )
            for agent_id in AGENT_IDS
        },
        "llm_reasoning_traces": reasoning_traces,
        "llm_usage": usage_records,
        "llm_usage_summary": summarize_llm_usage(usage_records),
        "analysis_metadata": {
            "controlled_bob_observed_verdict": bool(config.controlled_bob_verdict)
            and config.controlled_bob_observed_verdict,
            "reward": config.reward,
            "verdict_review": config.verdict_review,
            "verdict_evaluation": config.verdict_evaluation,
            "reflection": config.reflection,
            "reward_by_agent": outcomes.rewards,
            # Include the reward scheme so episode rows can be interpreted independently.
            "reward_scheme": config.reward_scheme.label,
            "reward_sharing": config.reward_scheme.sharing,
            "reward_terms": config.reward_scheme.terms,
            "verdict_correct_by_agent": {
                agent_id: outcomes.agents[agent_id].verdict_correct
                for agent_id in AGENT_IDS
            },
            "verdict_by_agent": {
                agent_id: outcomes.agents[agent_id].verdict for agent_id in AGENT_IDS
            },
            "expected_verdict_by_agent": {
                agent_id: evaluations[agent_id].expected_verdict
                for agent_id in AGENT_IDS
            },
            "cross_episode_memory_scope": config.cross_episode_memory_scope,
            "policy_prompt_by_agent": {
                agent_id: verdict_policy_section(
                    config.verdict_policy,
                    display_name(peer_id(agent_id)),
                )
                for agent_id in AGENT_IDS
            },
        },
        "channel_transcript": _build_channel_transcript(events=state["events"]),
        "events": state["events"],
    }


def build_run_output(
    args: argparse.Namespace,
    manifest_path: Path,
    pair_count: int,
    initial_verdict_policy: str,
    verdict_policy_by_episode: list[str],
    throttle_policy_by_episode: list[str],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the complete serializable state of a run."""
    reward_scheme = reward_scheme_from_args(args)
    system_prompts = {
        agent_id: initial_agent_messages(
            agent_id=agent_id,
            max_rounds=5
            if agent_id == "bob" and args.bob_model == "controlled"
            else args.max_rounds,
            reward_objective=args.reward
            and not (agent_id == "bob" and args.bob_model == "controlled"),
            reward_scheme=reward_scheme,
        )[0]["content"]
        for agent_id in AGENT_IDS
    }
    llm_usage_records = [
        usage for result in results for usage in result.get("llm_usage", [])
    ]
    return {
        "run_config": {
            **{
                key: getattr(args, key, "")
                for key in (
                    "controlled_bob_task_model",
                    "controlled_bob_verdict",
                    "controlled_bob_cache",
                    "controlled_bob_messages",
                )
            },
            "controlled_bob_observed_verdict": getattr(args, "controlled_bob_observed_verdict", False),
            "manifest": str(manifest_path),
            "pair_count": pair_count,
            # Record each agent's model independently.
            "models": {
                agent_id: getattr(args, f"{agent_id}_model") for agent_id in AGENT_IDS
            },
            "endpoints": {
                agent_id: get_endpoint_metadata(
                    args.controlled_bob_task_model
                    if agent_id == "bob" and args.bob_model == "controlled"
                    else getattr(args, f"{agent_id}_model")
                )
                for agent_id in AGENT_IDS
            },
            "litellm_version": get_litellm_version(),
            # Requested settings before LiteLLM applies provider-specific mappings and defaults.
            "llm_request_parameters": {
                agent_id: {
                    "temperature": getattr(args, f"{agent_id}_temperature"),
                    "max_output_tokens": getattr(
                        args,
                        f"{agent_id}_max_output_tokens",
                    ),
                    "reasoning_effort": getattr(
                        args,
                        f"{agent_id}_reasoning_effort",
                    ),
                }
                for agent_id in AGENT_IDS
            },
            "reasoning_efforts": {
                agent_id: getattr(args, f"{agent_id}_reasoning_effort")
                for agent_id in AGENT_IDS
            },
            "llm_usage_journal": {
                "path": LLM_USAGE_JOURNAL_FILENAME,
                "format": "jsonl",
                "version": 1,
                "includes_failed_attempts": True,
            },
            "max_rounds": args.max_rounds,
            "task_attempts": TASK_ATTEMPTS,
            "communication_attempts": COMMUNICATION_ATTEMPTS,
            "verdict_attempts": VERDICT_ATTEMPTS,
            "tool_choice_mode": "required",
            "throttle_spans": args.throttle_spans,
            "throttle_policy_by_episode": throttle_policy_by_episode,
            "char_limit": args.char_limit,
            "memory_scope": MEMORY_SCOPE,
            "cross_episode_memory_scope": args.cross_episode_memory_scope,
            "cross_episode_memory_length": {
                agent_id: getattr(args, f"{agent_id}_cross_episode_memory_length")
                for agent_id in AGENT_IDS
            },
            "verdict_policy": args.verdict_policy,
            "initial_verdict_policy": initial_verdict_policy,
            "verdict_policy_spans": args.verdict_policy_spans,
            "verdict_policy_by_episode": verdict_policy_by_episode,
            "run_label": args.run_label,
            "episode_count": len(results),
            "reflection": args.reflection,
            "reward": args.reward,
            "verdict_review": args.verdict_review,
            "verdict_evaluation": reward_scheme.verdict_review_direction(args.verdict_review),
            "reward_scope": args.reward_scope,
            "reward_type": args.reward_type,
            "reward_scheme": reward_scheme.label,
            "reward_sharing": reward_scheme.sharing,
            "reward_terms": reward_scheme.terms,
            "prompt_texts": {
                "system_prompts": system_prompts,
                "reward_objective_texts": {
                    agent_id: reward_objective_text(
                        args.reward
                        and not (agent_id == "bob" and args.bob_model == "controlled"),
                        display_name(peer_id(agent_id)),
                        reward_scheme,
                    )
                    for agent_id in AGENT_IDS
                },
            },
        },
        "llm_usage_summary": summarize_llm_usage(llm_usage_records),
        "results": results,
    }


def write_run_output(output_path: Path, run_output: dict[str, Any]) -> None:
    """Atomically write a run snapshot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(run_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
