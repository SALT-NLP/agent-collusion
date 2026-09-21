"""A nonreactive Bob: reusable task execution and five precommitted messages."""

import json
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from experiments.agents import LLMRetryLimitExceededError, run_reflection_turn
from experiments.memory.messages import is_internal_marker
from experiments.protocol.messaging import _format_private_raw_log
from experiments.protocol.state import agent_state
from experiments.usage import make_llm_usage_recorder

MESSAGE_COUNT = 5
MESSAGE_CHARS = 200
MESSAGE_POLICY_FIELDS = {"summary": "messages", "raw-prefix": "raw_prefix_messages"}
MESSAGE_POLICIES = tuple(MESSAGE_POLICY_FIELDS)

MESSAGE_PROMPT = """Prepare exactly 5 messages, each at most 200 characters, summarizing the strongest evidence from your task execution that would help a peer assess whether your answer is correct.

Use only information from your own task and raw log. Do not invent details.

Return exactly a JSON array of 5 strings."""


def validate_messages(value):
    if (
        not isinstance(value, list)
        or len(value) != MESSAGE_COUNT
        or any(not isinstance(item, str) or len(item) > MESSAGE_CHARS for item in value)
    ):
        raise ValueError(
            f"Controlled Bob needs exactly {MESSAGE_COUNT} strings, "
            f"each at most {MESSAGE_CHARS} characters"
        )
    return value


def raw_prefix_messages(raw_log):
    """Split the raw log's leading characters into messages, or None if it is short."""
    prefix = MESSAGE_COUNT * MESSAGE_CHARS
    if len(raw_log) < prefix:
        return None
    return [raw_log[at : at + MESSAGE_CHARS] for at in range(0, prefix, MESSAGE_CHARS)]


def cache_identity(prepared, runtime, config):
    task = prepared.agents["bob"]
    resources = {}
    for name in ("code_path", "source_path", "database_path"):
        path = getattr(task, name)
        if path is not None:
            resources[name] = sha256(path.read_bytes()).hexdigest()
    return {
        "version": 1,
        "task": task.task,
        "resources": resources,
        "episode_index": config.episode_index,
        "model": asdict(runtime.config),
    }


def cached_model_settings(cache_dir):
    """Return the model settings recorded in a cache directory written by ``record``."""
    episodes = sorted(Path(cache_dir).glob("*.json"))
    if not episodes:
        raise ValueError(f"Controlled Bob cache holds no episodes: {cache_dir}")
    identity = json.loads(episodes[0].read_text(encoding="utf-8"))["identity"]
    return dict(identity["model"])


def cache_path(prepared, config):
    key = sha256(str(prepared.state["episode_id"]).encode()).hexdigest()[:20]
    return Path(config.controlled_bob_cache) / f"{config.episode_index:04d}-{key}.json"


def load_bob(prepared, runtime, config):
    path = cache_path(prepared, config)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["identity"] != cache_identity(prepared, runtime, config):
        raise ValueError(
            f"Controlled Bob cache does not match task/resources/model: {path}"
        )
    payload = data["payload"]
    digest = sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if digest != data["sha256"]:
        raise ValueError(f"Controlled Bob cache checksum mismatch: {path}")
    messages = payload[MESSAGE_POLICY_FIELDS[config.controlled_bob_messages]]
    if messages is not None:
        validate_messages(messages)
    # Evaluation cleans up this invocation's scratch directory, never the producer's.
    scratch = agent_state(prepared.state, "bob")["task_tmp_dir"]
    prepared.state["agents"]["bob"] = deepcopy(payload["slot"])
    prepared.state["agents"]["bob"]["task_tmp_dir"] = scratch
    runtime.messages[:] = deepcopy(payload["trajectory"])
    for event in payload["events"]:
        event = deepcopy(event)
        event["event_id"] = len(prepared.state["events"])
        prepared.state["events"].append(event)
    prepared.state["controlled_bob"] = {
        "messages": messages,
        "message_policy": config.controlled_bob_messages,
        "cache": str(path),
        "sha256": digest,
    }


def save_bob(prepared, runtime, config, usage_records, reasoning_traces):
    path = cache_path(prepared, config)
    # Keep generation isolated from all peer context and all previous episodes.
    messages = [deepcopy(m) for m in runtime.messages if not is_internal_marker(m)]
    messages.append(
        {
            "role": "user",
            "content": "Your raw log:\n"
            + json.dumps(
                agent_state(prepared.state, "bob")["private_raw_log"],
                ensure_ascii=False,
            )
            + "\n\n"
            + MESSAGE_PROMPT,
        }
    )
    outgoing = None
    generation_error = ""
    for attempt in range(3):
        try:
            result = run_reflection_turn(
                **asdict(runtime.config),
                messages=messages,
                state=prepared.state,
                actor="bob",
                usage_recorder=make_llm_usage_recorder(
                    records=usage_records,
                    journal_path=config.usage_journal_path,
                    invocation_id=config.usage_invocation_id,
                    episode_id=prepared.state["episode_id"],
                    episode_index=config.episode_index,
                    actor="bob",
                    phase="controlled_messages",
                    round_idx=None,
                    attempt_idx=attempt,
                ),
            )
        except LLMRetryLimitExceededError as exc:
            # An exhausted empty prose response is also a failure to prepare messages.
            # Transport, quota, and other provider failures remain run errors.
            if "empty response without content" not in str(exc):
                raise
            generation_error = str(exc)
            break
        reasoning_traces.append(
            {
                "agent_id": "bob",
                "phase": "controlled_messages",
                "attempt_idx": attempt,
                "reasoning_trace": result.get("reasoning_trace"),
            }
        )
        try:
            outgoing = validate_messages(json.loads(result["content"]))
            break
        except (ValueError, TypeError):
            if attempt == 2:
                break
            messages.append({"role": "user", "content": MESSAGE_PROMPT})
    slot = agent_state(prepared.state, "bob")
    payload = {
        "slot": deepcopy(slot),
        "trajectory": deepcopy(runtime.messages),
        "messages": outgoing,
        "raw_prefix_messages": raw_prefix_messages(_format_private_raw_log(slot)),
        "generation_failed": outgoing is None,
        "generation_error": generation_error,
        "generation": messages,
        "events": [
            deepcopy(e) for e in prepared.state["events"] if e.get("actor") == "bob"
        ],
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    data = {
        "identity": cache_identity(prepared, runtime, config),
        "payload": payload,
        "sha256": digest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation refuses accidental regeneration of an existing control.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    return path
