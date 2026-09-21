"""Run one episode through task, communication, and verdict phases.

Both agents cross phase boundaries together. Communication alternates in
AGENT_IDS order, with one turn per agent per round.
"""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from litellm.exceptions import ContextWindowExceededError

from experiments.controlled import load_bob
from experiments.protocol.messaging import _send_message
from experiments.protocol.submissions import _submit_verdict
from experiments.agents import run_agent_turn, run_reflection_turn
from experiments.evaluation import evaluate_saved_artifact
from experiments.memory.cross_episode import (
    append_agent_outcome_feedback,
    append_outcome_feedback_boundary,
    append_reflection,
)
from experiments.memory.messages import (
    active_episode_start_message,
    communication_phase_start_message,
    drop_oldest_episode_block,
)
from experiments.models import (
    AgentConfig,
    AgentOutcome,
    AgentRuntime,
    TaskEvaluation,
    EpisodeOutcomes,
    EpisodeRunConfig,
    PreparedAgentTask,
    PreparedPair,
)
from experiments.output import build_episode_result
from experiments.prompts.messages import (
    AUTO_FORCED_REJECT_REASONING,
    agent_outcome_feedback_message,
    agent_reflection_prompt,
    communication_round_instruction,
    communication_round_tag,
    incoming_channel_message,
    peer_delivery_failure_notice,
)
from experiments.prompts.system import initial_agent_messages
from experiments.prompts.tasks import (
    own_code_task_message,
    own_data_search_task_message,
    own_extraction_task_message,
    judging_code_task_message,
    judging_data_search_task_message,
    judging_extraction_task_message,
    verdict_phase_message,
)
from experiments.protocol import (
    COMMUNICATION_ATTEMPTS,
    TASK_ATTEMPTS,
    VERDICT_ATTEMPTS,
)
from experiments.protocol.rewards import episode_rewards
from experiments.protocol.state import (
    AGENT_IDS,
    agent_state,
    create_channel_state,
    deliver_runner_notice,
    peer_id,
    pop_incoming_messages,
    reset_message_delivery,
    set_phase,
    verdict_is_correct,
)
from experiments.tasks import (
    task_file_path,
    task_type_of,
    display_task_id,
    is_code_task_type,
    sanitize_label,
    validate_task_pair,
)
from experiments.tool_schemas import FINAL_VERDICT_TOOL_NAME, forced_tool_choice
from experiments.usage import make_llm_usage_recorder

TEMP_TASK_PREFIX_MAX_LENGTH = 80

_PHASE_BUDGETS = {
    "task": TASK_ATTEMPTS,
    "communication": COMMUNICATION_ATTEMPTS,
    "verdict": VERDICT_ATTEMPTS,
}


def _inject_incoming_messages(
    messages: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    max_rounds: int,
) -> None:
    """Deliver queued messages with the round in which they were written.

    An empty inbox adds nothing; ``max_rounds`` is only the tag denominator.
    """
    if not incoming:
        return

    parts = []
    for item in incoming:
        round_tag = communication_round_tag(item["round"] + 1, max_rounds)
        parts.append(
            incoming_channel_message(
                sender=item["from"],
                message_type=item.get("message_type", "other"),
                content=item["content"],
                round_tag=round_tag,
            )
        )
    messages.append({"role": "user", "content": "\n\n".join(parts)})


def _phase_completed(turn_result: dict[str, Any], phase: str) -> bool:
    """Check whether a successful tool result completed the current phase."""
    for call in turn_result["tool_calls"]:
        result = call["result"]
        if phase == "task":
            if call["tool_name"] == "get_log" and result.get("success") is True:
                return True
        elif phase == "communication":
            if call["tool_name"] == "send_message" and result.get("success") is True:
                return True
        elif phase == "verdict":
            # Use the schema name to recognize a successful verdict submission.
            if (
                call["tool_name"] == FINAL_VERDICT_TOOL_NAME
                and result.get("recorded") is True
            ):
                return True
        else:
            raise ValueError(f"Unknown phase: {phase}")
    return False


def _spent_an_attempt(turn_result: dict[str, Any], phase: str) -> bool:
    """Count every model turn as an attempt, including missing or rejected tool calls."""
    if phase not in _PHASE_BUDGETS:
        raise ValueError(f"Unknown phase: {phase}")
    return True


def _has_verdict(state: dict[str, Any], agent_id: str) -> bool:
    return agent_state(state, agent_id)["verdict_submission"] is not None


def log_progress(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


def _summarize_turn(turn_result: dict[str, Any]) -> str:
    parts: list[str] = []
    for call in turn_result.get("tool_calls", []):
        name = call.get("tool_name", "?")
        res = call.get("result") or {}
        suffix = ""
        # Measure test output length; get_log already reports its payload size.
        if isinstance(res.get("raw_log_chars"), int):
            suffix = f", raw_log_chars={res['raw_log_chars']}"
        elif isinstance(res.get("toolchain_output"), str):
            suffix = f", toolchain_output_chars={len(res['toolchain_output'])}"
        if res.get("success") is True or res.get("recorded") is True:
            parts.append(f"{name}(ok{suffix})")
        elif res.get("error"):
            parts.append(f"{name}(err:{str(res.get('error'))[:60]})")
        else:
            parts.append(f"{name}({res})")
    return ", ".join(parts) if parts else "(no tool calls)"


def _append_llm_reasoning_trace(
    traces: list[dict[str, Any]],
    *,
    agent_id: str,
    phase: str,
    round_idx: int | None,
    attempt_idx: int | None,
    turn_result: dict[str, Any],
) -> None:
    trace = turn_result.get("reasoning_trace")
    if not trace:
        return
    traces.append(
        {
            "agent_id": agent_id,
            "phase": phase,
            "round": round_idx,
            "attempt": attempt_idx,
            "raw_response_model": turn_result.get("raw_response_model"),
            "reasoning_trace": deepcopy(trace),
            "reasoning_trace_chars": turn_result.get("reasoning_trace_chars", 0),
        }
    )


def _is_context_window_error(exc: Exception) -> bool:
    if isinstance(exc, ContextWindowExceededError):
        return True
    text = str(exc).lower()
    return (
        "context_length_exceeded" in text
        or "contextwindowexceeded" in text
        or "input tokens exceed" in text
        or "maximum context" in text
        or "maximum prompt length" in text
        or ("request contains" in text and "maximum prompt" in text)
    )


def _drop_oldest_cross_episode_block(
    messages: list[dict[str, Any]] | None,
    *,
    agent_id: str,
    verbose: bool,
    reason: str,
) -> bool:
    removed_messages = (
        drop_oldest_episode_block(messages) if messages is not None else 0
    )
    if removed_messages:
        log_progress(
            "    [context limit: dropped oldest cross-episode memory block "
            f"({agent_id} messages={removed_messages}) "
            f"and retrying {reason}]",
            verbose,
        )
        return True
    return False


def _run_agent_turn_with_context_pruning(
    *,
    runtime: AgentRuntime,
    state: dict[str, Any],
    tool_choice: str | dict[str, Any],
    verbose: bool,
    usage_recorder: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    while True:
        try:
            return run_agent_turn(
                actor=runtime.agent_id,
                model=runtime.config.model,
                reasoning_effort=runtime.config.reasoning_effort,
                temperature=runtime.config.temperature,
                max_output_tokens=runtime.config.max_output_tokens,
                messages=runtime.messages,
                state=state,
                tool_choice=tool_choice,
                usage_recorder=usage_recorder,
            )
        except Exception as exc:
            if not _is_context_window_error(exc):
                raise
            if not _drop_oldest_cross_episode_block(
                runtime.messages,
                agent_id=runtime.agent_id,
                verbose=verbose,
                reason=f"{runtime.agent_id} turn",
            ):
                raise


def _run_reflection_turn_with_context_pruning(
    *,
    runtime: AgentRuntime,
    state: dict[str, Any],
    prompt: str,
    verbose: bool,
    usage_recorder: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    while True:
        reflection_messages = deepcopy(runtime.messages)
        reflection_messages.append({"role": "user", "content": prompt})
        try:
            return run_reflection_turn(
                model=runtime.config.model,
                reasoning_effort=runtime.config.reasoning_effort,
                temperature=runtime.config.temperature,
                max_output_tokens=runtime.config.max_output_tokens,
                messages=reflection_messages,
                state=state,
                actor=runtime.agent_id,
                usage_recorder=usage_recorder,
            )
        except Exception as exc:
            if not _is_context_window_error(exc):
                raise
            if not _drop_oldest_cross_episode_block(
                runtime.messages,
                agent_id=runtime.agent_id,
                verbose=verbose,
                reason=f"{runtime.agent_id} reflection",
            ):
                raise


def _run_reflection(
    *,
    runtime: AgentRuntime,
    state: dict[str, Any],
    prompt: str,
    verbose: bool,
    reasoning_traces: list[dict[str, Any]],
    usage_recorder: Callable[[dict[str, Any]], None] | None,
) -> tuple[str, dict[str, Any]]:
    """Return reflection text and replayable reasoning for persistence in agent history."""
    agent_id = runtime.agent_id
    log_progress(f"  {agent_id} outcome reflection start (private)", verbose)
    result = _run_reflection_turn_with_context_pruning(
        runtime=runtime,
        state=state,
        prompt=prompt,
        verbose=verbose,
        usage_recorder=usage_recorder,
    )
    _append_llm_reasoning_trace(
        reasoning_traces,
        agent_id=agent_id,
        phase="private_reflection",
        round_idx=None,
        attempt_idx=None,
        turn_result=result,
    )
    reflection = str(result.get("content", "")).strip()
    log_progress(
        f"  {agent_id} outcome reflection chars={len(reflection)}",
        verbose,
    )
    return reflection, dict(result.get("replayed_reasoning") or {})


def _temporary_task_dir_name(
    *,
    task_id: str,
    temp_run_id: str,
    episode_index: int,
    agent_id: str,
) -> str:
    """Create a test directory isolated by episode and agent."""
    identity = "\0".join((task_id, temp_run_id, str(episode_index), agent_id))
    digest = sha256(identity.encode("utf-8")).hexdigest()[:16]
    prefix = sanitize_label(f"{task_id}_{agent_id}")[:TEMP_TASK_PREFIX_MAX_LENGTH]
    prefix = prefix.rstrip("_.-") or "task"
    return f"{prefix}_ep{episode_index:03d}_{digest}"


def _prepare_agent_task(
    *,
    agent_id: str,
    task: dict[str, Any],
    task_type: str,
    state: dict[str, Any],
    repo_root: Path,
    config: EpisodeRunConfig,
) -> PreparedAgentTask:
    """Prepare task, judging, and verdict openings.

    Only the task owner receives resource paths; the judging brief omits them.
    """
    slot = agent_state(state, agent_id)
    # Use a shared, 1-based episode number in agent-facing prompts.
    episode_number = config.episode_index + 1
    task_id = str(task["task_id"])
    display_id = display_task_id(task)
    slot["display_id"] = display_id
    verdict_prompt = verdict_phase_message(episode_number=episode_number, peer=agent_id)

    if is_code_task_type(task_type):
        code_path = task_file_path(task, "code_path", repo_root)
        function_name = str(task["function_name"])
        slot["target_code_path"] = str(code_path)
        slot["function_name"] = function_name
        slot["task_tmp_dir"] = str(
            repo_root
            / "results"
            / "tmp_tests"
            / _temporary_task_dir_name(
                task_id=task_id,
                temp_run_id=config.temp_run_id,
                episode_index=config.episode_index,
                agent_id=agent_id,
            )
        )
        return PreparedAgentTask(
            task=task,
            display_id=display_id,
            code_path=code_path,
            function_name=function_name,
            own_task_prompt=own_code_task_message(
                episode_number=episode_number,
                display_id=display_id,
                code_path=str(code_path.relative_to(repo_root)),
                function_name=function_name,
            ),
            judging_prompt=judging_code_task_message(
                peer=agent_id,
                episode_number=episode_number,
                display_id=display_id,
                function_name=function_name,
                verdict_policy=config.verdict_policy,
            ),
            verdict_prompt=verdict_prompt,
        )

    if task_type == "record_extraction":
        source_path = task_file_path(task, "source_path", repo_root)
        slot["source_path"] = str(source_path)
        instruction = str(task["extraction_instruction"])
        return PreparedAgentTask(
            task=task,
            display_id=display_id,
            source_path=source_path,
            own_task_prompt=own_extraction_task_message(
                episode_number=episode_number,
                display_id=display_id,
                source_path=str(source_path.relative_to(repo_root)),
                extraction_instruction=instruction,
            ),
            judging_prompt=judging_extraction_task_message(
                peer=agent_id,
                episode_number=episode_number,
                display_id=display_id,
                extraction_instruction=instruction,
                verdict_policy=config.verdict_policy,
            ),
            verdict_prompt=verdict_prompt,
        )

    if task_type == "data_search":
        database_path = task_file_path(task, "database_path", repo_root)
        slot["database_path"] = str(database_path)
        answer_key = [str(column) for column in task.get("answer_key", [])]
        slot["answer_key"] = answer_key
        search_prompt = str(task["search_prompt"])
        return PreparedAgentTask(
            task=task,
            display_id=display_id,
            database_path=database_path,
            own_task_prompt=own_data_search_task_message(
                episode_number=episode_number,
                display_id=display_id,
                database_path=str(database_path.relative_to(repo_root)),
                search_prompt=search_prompt,
                answer_key=answer_key,
            ),
            judging_prompt=judging_data_search_task_message(
                peer=agent_id,
                episode_number=episode_number,
                display_id=display_id,
                search_prompt=search_prompt,
                answer_key=answer_key,
                verdict_policy=config.verdict_policy,
            ),
            verdict_prompt=verdict_prompt,
        )

    raise ValueError(f"Unknown task_type: {task_type}")


def _prepare_pair(
    *,
    pair: tuple[dict[str, Any], dict[str, Any]],
    repo_root: Path,
    config: EpisodeRunConfig,
) -> PreparedPair:
    """Validate a task pair and prepare both agent slots for the episode."""
    first, second = pair
    validate_task_pair(first, second)
    task_type = task_type_of(first)
    state = create_channel_state(
        episode_id=f"{first['task_id']}+{second['task_id']}",
        throttled=config.throttled,
        char_limit=config.char_limit,
    )
    state.update(
        {
            "repo_root": str(repo_root),
            "task_type": task_type,
        }
    )
    prepared = PreparedPair(task_type=task_type, state=state)
    for agent_id, task in zip(AGENT_IDS, (first, second)):
        prepared.agents[agent_id] = _prepare_agent_task(
            agent_id=agent_id,
            task=task,
            task_type=task_type,
            state=state,
            repo_root=repo_root,
            config=config,
        )
    return prepared


def _open_episode(
    messages: list[dict[str, Any]],
    own_task_prompt: str,
) -> None:
    """Open the task phase with the agent's own task and an internal episode marker.

    The peer task and verdict policy are introduced only when communication opens.
    """
    messages.append(active_episode_start_message())
    messages.append({"role": "user", "content": own_task_prompt})


def _append_episode_open_context(
    *,
    prepared: PreparedPair,
    agent_messages: dict[str, list[dict[str, Any]]],
) -> None:
    for agent_id, messages in agent_messages.items():
        _open_episode(messages, prepared.agents[agent_id].own_task_prompt)


def _run_agent_attempts(
    *,
    runtime: AgentRuntime,
    state: dict[str, Any],
    phase: str,
    tool_choice: str | dict[str, Any],
    episode_id: str,
    round_index: int | None,
    reasoning_traces: list[dict[str, Any]],
    usage_records: list[dict[str, Any]],
    config: EpisodeRunConfig,
) -> bool:
    """Run one agent until the phase completes or its attempt budget is exhausted."""
    agent_id = runtime.agent_id
    attempts_left = _PHASE_BUDGETS[phase]
    attempt_index = 0
    while attempts_left > 0:
        reset_message_delivery(state, agent_id)
        log_progress(
            f"    {agent_id} → LLM ({phase}, {attempts_left} left) …",
            config.verbose,
        )
        turn = _run_agent_turn_with_context_pruning(
            runtime=runtime,
            state=state,
            tool_choice=tool_choice,
            verbose=config.verbose,
            usage_recorder=make_llm_usage_recorder(
                records=usage_records,
                journal_path=config.usage_journal_path,
                invocation_id=config.usage_invocation_id,
                episode_id=episode_id,
                episode_index=config.episode_index,
                actor=agent_id,
                phase=phase,
                round_idx=round_index,
                attempt_idx=attempt_index,
            ),
        )
        _append_llm_reasoning_trace(
            reasoning_traces,
            agent_id=agent_id,
            phase=phase,
            round_idx=round_index,
            attempt_idx=attempt_index,
            turn_result=turn,
        )
        log_progress(f"    {agent_id} ← {_summarize_turn(turn)}", config.verbose)
        attempt_index += 1
        if _phase_completed(turn, phase):
            return True
        if _spent_an_attempt(turn, phase):
            attempts_left -= 1
    return False


def _run_task_phase(
    *,
    state: dict[str, Any],
    runtimes: dict[str, AgentRuntime],
    episode_id: str,
    config: EpisodeRunConfig,
    reasoning_traces: list[dict[str, Any]],
    usage_records: list[dict[str, Any]],
) -> None:
    """Run each agent's private task phase sequentially.

    Budget exhaustion ends the phase; an unsaved answer is evaluated as missing.
    """
    set_phase(state, "task")
    for agent_id in AGENT_IDS:
        if agent_id == "bob" and config.controlled_bob_verdict:
            continue
        log_progress(f"  Task phase: {agent_id} (episode={episode_id})", config.verbose)
        _run_agent_attempts(
            runtime=runtimes[agent_id],
            state=state,
            phase="task",
            tool_choice="required",
            episode_id=episode_id,
            round_index=None,
            reasoning_traces=reasoning_traces,
            usage_records=usage_records,
            config=config,
        )


def _open_communication_phase(
    *,
    prepared: PreparedPair,
    runtimes: dict[str, AgentRuntime],
) -> None:
    """Give both agents the peer task and verdict policy before communication round 1."""
    for agent_id, runtime in runtimes.items():
        # Place the memory boundary before the peer brief so reduced memory retains it.
        runtime.messages.append(communication_phase_start_message())
        runtime.messages.append(
            {
                "role": "user",
                "content": prepared.agents[peer_id(agent_id)].judging_prompt,
            }
        )


def _run_communication_phase(
    *,
    prepared: PreparedPair,
    state: dict[str, Any],
    runtimes: dict[str, AgentRuntime],
    episode_id: str,
    config: EpisodeRunConfig,
    reasoning_traces: list[dict[str, Any]],
    usage_records: list[dict[str, Any]],
) -> None:
    """Run ``config.max_rounds`` exchanges in AGENT_IDS order.

    Messages enter the peer inbox immediately and are read at its next turn.
    """
    set_phase(state, "communication")
    active_runtimes = {
        actor: runtime
        for actor, runtime in runtimes.items()
        if not (actor == "bob" and config.controlled_bob_verdict)
    }
    _open_communication_phase(prepared=prepared, runtimes=active_runtimes)
    # Force the communication phase's single tool by name.
    tool_choice = forced_tool_choice(prepared.task_type, "communication")
    for round_index in range(config.max_rounds):
        state["round"] = round_index
        log_progress(
            f"  Round {round_index + 1}/{config.max_rounds} start "
            f"(episode={episode_id})",
            config.verbose,
        )
        for agent_id in AGENT_IDS:
            runtime = runtimes[agent_id]
            if agent_id == "bob" and config.controlled_bob_verdict:
                pop_incoming_messages(state, "bob")
                reset_message_delivery(state, "bob")
                outgoing = state["controlled_bob"]["messages"]
                delivered = outgoing is not None and _send_message(
                    state, "bob", outgoing[round_index]
                ).get("success")
                if not delivered:
                    deliver_runner_notice(
                        state,
                        "alice",
                        peer_delivery_failure_notice(
                            peer="bob", round_number=round_index + 1
                        ),
                    )
                continue
            _inject_incoming_messages(
                runtime.messages,
                incoming=pop_incoming_messages(state, agent_id),
                max_rounds=config.max_rounds,
            )
            # Add one round instruction per turn, after incoming messages; retries reuse it.
            runtime.messages.append(
                {
                    "role": "user",
                    "content": communication_round_instruction(
                        round_index + 1, config.max_rounds
                    ),
                }
            )
            delivered = _run_agent_attempts(
                runtime=runtime,
                state=state,
                phase="communication",
                tool_choice=tool_choice,
                episode_id=episode_id,
                round_index=round_index,
                reasoning_traces=reasoning_traces,
                usage_records=usage_records,
                config=config,
            )
            # Queue a silence notice for the peer's next turn.
            if not delivered:
                deliver_runner_notice(
                    state,
                    peer_id(agent_id),
                    peer_delivery_failure_notice(
                        peer=agent_id,
                        round_number=round_index + 1,
                    ),
                )
    _close_communication_phase(state=state, runtimes=runtimes, config=config)


def _close_communication_phase(
    *,
    state: dict[str, Any],
    runtimes: dict[str, AgentRuntime],
    config: EpisodeRunConfig,
) -> None:
    """Deliver the final queued message or silence notice before the verdict phase."""
    for agent_id in AGENT_IDS:
        _inject_incoming_messages(
            runtimes[agent_id].messages,
            incoming=pop_incoming_messages(state, agent_id),
            max_rounds=config.max_rounds,
        )


def _record_forced_verdict(state: dict[str, Any], agent_id: str) -> None:
    """Assign reject to an agent that exhausted its attempts without a verdict."""
    slot = agent_state(state, agent_id)
    slot["verdict_submission"] = {
        "verdict": "reject",
        "reasoning": AUTO_FORCED_REJECT_REASONING,
        "target": peer_id(agent_id),
        "forced": True,
    }
    state["events"].append(
        {
            "event_id": len(state["events"]),
            "round": state["round"],
            "phase": state["phase"],
            "actor": "runner",
            "target": peer_id(agent_id),
            "on_behalf_of": agent_id,
            "tool": FINAL_VERDICT_TOOL_NAME,
            "success": True,
            "verdict": "reject",
        }
    )


def _open_verdict_phase(
    *,
    prepared: PreparedPair,
    runtimes: dict[str, AgentRuntime],
) -> None:
    """Open the verdict phase for both agents after draining communication inboxes."""
    for agent_id, runtime in runtimes.items():
        runtime.messages.append(
            {
                "role": "user",
                "content": prepared.agents[peer_id(agent_id)].verdict_prompt,
            }
        )


def _run_verdict_phase(
    *,
    prepared: PreparedPair,
    state: dict[str, Any],
    runtimes: dict[str, AgentRuntime],
    episode_id: str,
    config: EpisodeRunConfig,
    reasoning_traces: list[dict[str, Any]],
    usage_records: list[dict[str, Any]],
) -> None:
    """Collect both peer verdicts before scoring the episode."""
    set_phase(state, "verdict")
    active_runtimes = {
        actor: runtime
        for actor, runtime in runtimes.items()
        if not (actor == "bob" and config.controlled_bob_verdict)
    }
    _open_verdict_phase(prepared=prepared, runtimes=active_runtimes)
    tool_choice = forced_tool_choice(prepared.task_type, "verdict")
    for agent_id in AGENT_IDS:
        runtime = runtimes[agent_id]
        if agent_id == "bob" and config.controlled_bob_verdict:
            _submit_verdict(
                state,
                "bob",
                config.controlled_bob_verdict,
                "Fixed controlled-peer verdict.",
            )
            continue
        log_progress(
            f"  Verdict phase: {agent_id} (episode={episode_id})", config.verbose
        )
        _run_agent_attempts(
            runtime=runtime,
            state=state,
            phase="verdict",
            tool_choice=tool_choice,
            episode_id=episode_id,
            round_index=None,
            reasoning_traces=reasoning_traces,
            usage_records=usage_records,
            config=config,
        )
        if not _has_verdict(state, agent_id):
            _record_forced_verdict(state, agent_id)


def _process_episode_outcomes(
    *,
    episode_id: str,
    state: dict[str, Any],
    evaluations: dict[str, TaskEvaluation],
    runtimes: dict[str, AgentRuntime],
    config: EpisodeRunConfig,
    reasoning_traces: list[dict[str, Any]],
    usage_records: list[dict[str, Any]],
) -> EpisodeOutcomes:
    """Compute both rewards, deliver outcome feedback, and run enabled reflections."""
    verdicts = {
        agent_id: agent_state(state, agent_id)["verdict_submission"]
        for agent_id in AGENT_IDS
    }
    rewards = episode_rewards(
        scheme=config.reward_scheme,
        verdict_by_agent={
            agent_id: str(verdicts[agent_id]["verdict"]) for agent_id in AGENT_IDS
        },
        expected_verdict_by_agent={
            agent_id: evaluations[agent_id].expected_verdict for agent_id in AGENT_IDS
        },
    )
    if config.controlled_bob_verdict:
        rewards["bob"] = 0
    outcomes = EpisodeOutcomes()

    for agent_id in AGENT_IDS:
        peer = peer_id(agent_id)
        own_verdict = str(verdicts[agent_id]["verdict"])
        outcomes.agents[agent_id] = AgentOutcome(
            verdict=own_verdict,
            verdict_correct=verdict_is_correct(
                own_verdict,
                evaluations[peer].expected_verdict,
            ),
            reward=rewards[agent_id],
            verdict_reasoning=str(verdicts[agent_id].get("reasoning", "")),
            verdict_forced=bool(verdicts[agent_id].get("forced", False)),
        )

    # With neither reward feedback nor verdict review, add only the internal memory marker.
    for agent_id in AGENT_IDS:
        peer = peer_id(agent_id)
        if config.controlled_bob_verdict:
            append_outcome_feedback_boundary(runtimes[agent_id].messages)
            if agent_id == "bob":
                continue

            feedback = ""
            if config.reward or config.verdict_evaluation != "none":
                feedback = (
                    agent_outcome_feedback_message(
                        episode_number=config.episode_index + 1,
                        own_task_id=agent_state(state, agent_id)["display_id"],
                        peer_task_id=agent_state(state, peer)["display_id"],
                        peer=peer,
                        own_verdict=outcomes.agents[agent_id].verdict,
                        peer_expected_verdict=evaluations[peer].expected_verdict,
                        peer_verdict=outcomes.agents[peer].verdict,
                        own_expected_verdict=evaluations[agent_id].expected_verdict,
                        reward=config.reward,
                        verdict_evaluation=config.verdict_evaluation,
                        reward_scheme=config.reward_scheme,
                    )
                    + "\n\n"
                )
            if config.controlled_bob_observed_verdict:
                feedback += (
                    "## Bob's observed verdict\n\nBob submitted "
                    + outcomes.agents["bob"].verdict.upper()
                    + " on your task."
                )
            feedback = feedback.rstrip()
            if feedback:
                outcomes.agents[agent_id].feedback = feedback
                runtimes[agent_id].messages.append({"role": "user", "content": feedback})
            continue
        if not config.reward and config.verdict_evaluation == "none":
            append_outcome_feedback_boundary(runtimes[agent_id].messages)
            continue
        # Use display IDs in feedback to avoid exposing internal manifest identifiers.
        outcomes.agents[agent_id].feedback = append_agent_outcome_feedback(
            runtimes[agent_id].messages,
            episode_number=config.episode_index + 1,
            own_task_id=agent_state(state, agent_id)["display_id"],
            peer_task_id=agent_state(state, peer)["display_id"],
            peer=peer,
            own_verdict=outcomes.agents[agent_id].verdict,
            peer_expected_verdict=evaluations[peer].expected_verdict,
            peer_verdict=outcomes.agents[peer].verdict,
            own_expected_verdict=evaluations[agent_id].expected_verdict,
            reward=config.reward,
            verdict_evaluation=config.verdict_evaluation,
            reward_scheme=config.reward_scheme,
        )

    if config.reflection:
        prompt = agent_reflection_prompt()
        for agent_id in AGENT_IDS:
            if agent_id == "bob" and config.controlled_bob_verdict:
                continue
            outcome = outcomes.agents[agent_id]
            outcome.reflection_prompt = prompt
            outcome.reflection, replayed_reasoning = _run_reflection(
                runtime=runtimes[agent_id],
                state=state,
                prompt=prompt,
                verbose=config.verbose,
                reasoning_traces=reasoning_traces,
                usage_recorder=make_llm_usage_recorder(
                    records=usage_records,
                    journal_path=config.usage_journal_path,
                    invocation_id=config.usage_invocation_id,
                    episode_id=episode_id,
                    episode_index=config.episode_index,
                    actor=agent_id,
                    phase="private_reflection",
                    round_idx=None,
                    attempt_idx=None,
                ),
            )
            if outcome.reflection:
                append_reflection(
                    runtimes[agent_id].messages,
                    prompt=prompt,
                    reflection=outcome.reflection,
                    replayed_reasoning=replayed_reasoning,
                )

    # Report rewards per agent; separate schemes can pay different amounts.
    log_progress(
        "  Episode done: "
        + " ".join(
            f"{agent_id}(+{outcomes.agents[agent_id].reward} "
            f"{outcomes.agents[agent_id].verdict}"
            f"/{'right' if outcomes.agents[agent_id].verdict_correct else 'wrong'})"
            for agent_id in AGENT_IDS
        ),
        config.verbose,
    )
    return outcomes


def run_one_episode(
    *,
    pair: tuple[dict[str, Any], dict[str, Any]],
    agent_configs: dict[str, AgentConfig],
    repo_root: Path,
    config: EpisodeRunConfig,
    agent_messages: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if config.controlled_bob_verdict and config.controlled_bob_verdict not in (
        "accept",
        "reject",
    ):
        raise ValueError("Controlled Bob verdict must be accept or reject")
    prepared = _prepare_pair(pair=pair, repo_root=repo_root, config=config)
    state = prepared.state
    episode_id = str(state["episode_id"])
    llm_reasoning_traces: list[dict[str, Any]] = []
    llm_usage: list[dict[str, Any]] = []

    if agent_messages is None:
        agent_messages = {
            agent_id: initial_agent_messages(
                agent_id=agent_id,
                max_rounds=config.max_rounds,
                reward_objective=config.reward,
                reward_scheme=config.reward_scheme,
            )
            for agent_id in AGENT_IDS
        }

    if config.controlled_bob_verdict:
        if config.max_rounds not in (0, 5) or config.char_limit < 200:
            raise ValueError(
                "Controlled Bob requires 0 or 5 rounds and char_limit >= 200"
            )
        if not config.controlled_bob_cache:
            raise ValueError("Controlled Bob needs a cache directory")
        agent_messages["bob"] = initial_agent_messages(agent_id="bob", max_rounds=5)
    _append_episode_open_context(
        prepared=prepared,
        agent_messages=agent_messages,
    )
    runtimes = {
        agent_id: AgentRuntime(
            agent_id=agent_id,
            config=agent_configs[agent_id],
            messages=agent_messages[agent_id],
        )
        for agent_id in AGENT_IDS
    }

    phase_kwargs = {
        "state": state,
        "runtimes": runtimes,
        "episode_id": episode_id,
        "config": config,
        "reasoning_traces": llm_reasoning_traces,
        "usage_records": llm_usage,
    }
    if config.controlled_bob_verdict:
        load_bob(prepared, runtimes["bob"], config)
    _run_task_phase(**phase_kwargs)
    _run_communication_phase(prepared=prepared, **phase_kwargs)
    _run_verdict_phase(prepared=prepared, **phase_kwargs)

    evaluations = {
        agent_id: evaluate_saved_artifact(
            task=prepared.agents[agent_id].task,
            task_type=prepared.task_type,
            slot=agent_state(state, agent_id),
            repo_root=repo_root,
        )
        for agent_id in AGENT_IDS
    }
    outcomes = _process_episode_outcomes(
        episode_id=episode_id,
        state=state,
        evaluations=evaluations,
        runtimes=runtimes,
        config=config,
        reasoning_traces=llm_reasoning_traces,
        usage_records=llm_usage,
    )
    result = build_episode_result(
        prepared=prepared,
        evaluations=evaluations,
        outcomes=outcomes,
        repo_root=repo_root,
        config=config,
        reasoning_traces=llm_reasoning_traces,
        usage_records=llm_usage,
        agent_messages=agent_messages,
    )
    if config.controlled_bob_verdict:
        result["controlled_bob"] = {
            **state["controlled_bob"],
            "verdict": config.controlled_bob_verdict,
        }
    return result
