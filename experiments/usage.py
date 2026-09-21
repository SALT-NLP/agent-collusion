"""Persistent per-call token accounting and aggregate usage summaries."""

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isclose
from pathlib import Path
from typing import Any, Callable

import litellm

LLM_USAGE_JOURNAL_FILENAME = "llm_usage.jsonl"
TOKEN_FIELDS = (
    "input_tokens",
    "reasoning_tokens",
    "output_tokens",
    "provider_output_tokens",
    "total_tokens",
    "cached_input_tokens",
    # Partition input into cache reads, cache writes, and uncached tokens.
    "cache_write_input_tokens",
    "uncached_input_tokens",
    "billable_output_tokens",
)


@dataclass(frozen=True)
class ReportedUsage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    cache_write_input_tokens: int | None


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    return str(value)


def _first_int(mapping: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _nested_int(
    mapping: dict[str, Any],
    container_keys: tuple[str, ...],
    value_keys: tuple[str, ...],
) -> int | None:
    for container_key in container_keys:
        container = mapping.get(container_key)
        if not isinstance(container, dict):
            continue
        value = _first_int(container, *value_keys)
        if value is not None:
            return value
    return None


def _reported_usage(raw_usage: dict[str, Any]) -> ReportedUsage:
    reasoning_tokens = _first_int(
        raw_usage,
        "reasoning_tokens",
        "thoughts_token_count",
        "thoughtsTokenCount",
    )
    if reasoning_tokens is None:
        reasoning_tokens = _nested_int(
            raw_usage,
            ("completion_tokens_details", "output_tokens_details"),
            ("reasoning_tokens", "reasoningTokens"),
        )

    cached_input_tokens = _first_int(
        raw_usage,
        "cached_content_token_count",
        "cachedContentTokenCount",
    )
    if cached_input_tokens is None:
        cached_input_tokens = _nested_int(
            raw_usage,
            ("prompt_tokens_details", "input_tokens_details"),
            ("cached_tokens", "cachedTokens"),
        )

    cache_write_input_tokens = _first_int(
        raw_usage,
        "cache_creation_input_tokens",
        "cacheWriteInputTokens",
    )
    if cache_write_input_tokens is None:
        cache_write_input_tokens = _nested_int(
            raw_usage,
            ("prompt_tokens_details", "input_tokens_details"),
            ("cache_creation_tokens", "cacheCreationTokens"),
        )

    return ReportedUsage(
        input_tokens=_first_int(
            raw_usage,
            "prompt_tokens",
            "input_tokens",
            "prompt_token_count",
            "promptTokenCount",
        ),
        output_tokens=_first_int(
            raw_usage,
            "completion_tokens",
            "output_tokens",
            "candidates_token_count",
            "candidatesTokenCount",
        ),
        total_tokens=_first_int(
            raw_usage,
            "total_tokens",
            "total_token_count",
            "totalTokenCount",
        ),
        reasoning_tokens=reasoning_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
    )


def _output_token_breakdown(
    reported: ReportedUsage,
    requested_model: str,
) -> tuple[int | None, int | None, str, bool | None]:
    reasoning_tokens = reported.reasoning_tokens
    if reasoning_tokens is not None:
        if reported.output_tokens is None:
            return None, reasoning_tokens, "explicit", None
        inclusive_total = (
            reported.input_tokens + reported.output_tokens
            if reported.input_tokens is not None
            else None
        )
        separate_total = (
            inclusive_total + reasoning_tokens if inclusive_total is not None else None
        )
        if (
            reported.total_tokens is not None
            and reported.total_tokens == separate_total
        ):
            return reported.output_tokens, reasoning_tokens, "explicit", False
        return (
            max(reported.output_tokens - reasoning_tokens, 0),
            reasoning_tokens,
            "explicit",
            True,
        )

    if None in (
        reported.input_tokens,
        reported.output_tokens,
        reported.total_tokens,
    ):
        return None, None, "unavailable", None

    remainder = reported.total_tokens - reported.input_tokens - reported.output_tokens
    if remainder > 0:
        return reported.output_tokens, remainder, "inferred_from_total", False
    deployment = requested_model.rsplit("/", 1)[-1].lower()
    known_reasoning_model = any(
        name in deployment for name in ("gpt-5", "gemini", "qwen", "gemma")
    )
    if remainder == 0 and not known_reasoning_model:
        return reported.output_tokens, 0, "not_reported_assumed_zero", False
    return None, None, "unavailable", None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _price_from_cost_map(response: Any, requested_model: str) -> float | None:
    """Estimate list-rate cost using the response model, then the requested model."""
    for model in (None, requested_model):
        try:
            computed_cost = litellm.completion_cost(
                completion_response=response,
                **({} if model is None else {"model": model}),
            )
        except Exception:
            continue
        if _is_number(computed_cost):
            return float(computed_cost)
    return None


def _call_cost_usd(
    response: Any,
    requested_model: str,
) -> tuple[float | None, str, float | None]:
    """Resolve call cost and its source, retaining a differing list-rate estimate.

    A proxy-reported cost can differ from LiteLLM's published-rate estimate.
    Unpriced calls remain unavailable rather than being reported as zero.
    """
    from_map = _price_from_cost_map(response, requested_model)
    hidden_params = _get_attr(response, "_hidden_params", {})
    reported = None
    if isinstance(hidden_params, dict) and _is_number(
        hidden_params.get("response_cost")
    ):
        reported = float(hidden_params["response_cost"])

    if reported is None:
        if from_map is None:
            return None, "unavailable", None
        return from_map, "litellm_cost_map", None
    if from_map is None:
        # No list-rate estimate exists for this deployment.
        return reported, "endpoint_reported", None
    if isclose(reported, from_map, rel_tol=1e-9):
        return from_map, "litellm_cost_map", None
    return reported, "endpoint_reported", from_map


def normalize_llm_usage(
    response: Any,
    requested_model: str,
    *,
    call_failed: bool = False,
) -> dict[str, Any]:
    """Normalize provider usage while retaining original fields.

    Distinguish failed requests from successful responses with missing usage.
    """
    raw_usage = _jsonable(_get_attr(response, "usage", {}))
    if not isinstance(raw_usage, dict):
        raw_usage = {}
    reported = _reported_usage(raw_usage)
    (
        output_tokens,
        reasoning_tokens,
        reasoning_source,
        provider_output_includes_reasoning,
    ) = _output_token_breakdown(reported, requested_model)

    billable_output_tokens = None
    if reported.input_tokens is not None and reported.total_tokens is not None:
        billable_output_tokens = max(
            reported.total_tokens - reported.input_tokens,
            0,
        )
    elif reported.output_tokens is not None:
        billable_output_tokens = reported.output_tokens
    uncached_input_tokens = None
    if reported.input_tokens is not None:
        # LiteLLM includes cache reads and writes in total input tokens.
        uncached_input_tokens = max(
            reported.input_tokens
            - (reported.cached_input_tokens or 0)
            - (reported.cache_write_input_tokens or 0),
            0,
        )
    reported_no_tokens = (
        reported.input_tokens is None and reported.output_tokens is None
    )
    cost_usd_list_rate = None
    if reported_no_tokens and call_failed:
        # Record rejected or rate-limited requests as zero-cost failures.
        cost_usd, cost_usd_source = 0.0, "no_tokens_billed"
    elif reported_no_tokens:
        cost_usd, cost_usd_source = None, "unavailable"
    else:
        cost_usd, cost_usd_source, cost_usd_list_rate = _call_cost_usd(
            response,
            requested_model,
        )

    return {
        "requested_model": requested_model,
        "response_model": _get_attr(response, "model", requested_model),
        "input_tokens": reported.input_tokens,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
        "provider_output_tokens": reported.output_tokens,
        "total_tokens": reported.total_tokens,
        "cached_input_tokens": reported.cached_input_tokens,
        "cache_write_input_tokens": reported.cache_write_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "billable_output_tokens": billable_output_tokens,
        "reasoning_tokens_source": reasoning_source,
        "provider_output_includes_reasoning": provider_output_includes_reasoning,
        "cost_usd": cost_usd,
        "cost_usd_source": cost_usd_source,
        # Keep the list-rate estimate when it differs from the reported price.
        "cost_usd_list_rate": cost_usd_list_rate,
        "raw_usage": raw_usage,
    }


def _append_llm_usage_journal(
    journal_path: Path,
    record: dict[str, Any],
) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(record)
    payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
    with journal_path.open("a", encoding="utf-8") as journal:
        journal.write(json.dumps(payload, ensure_ascii=False) + "\n")
        journal.flush()
        os.fsync(journal.fileno())


def make_llm_usage_recorder(
    *,
    records: list[dict[str, Any]],
    journal_path: Path | None,
    invocation_id: str,
    episode_id: str,
    episode_index: int,
    actor: str,
    phase: str,
    round_idx: int | None,
    attempt_idx: int | None,
) -> Callable[[dict[str, Any]], None]:
    def record_usage(usage: dict[str, Any]) -> None:
        episode_record = {
            "episode_id": episode_id,
            "actor": actor,
            "phase": phase,
            "round": None if round_idx is None else round_idx + 1,
            "agent_attempt": None if attempt_idx is None else attempt_idx + 1,
            **deepcopy(usage),
        }
        records.append(episode_record)
        if journal_path is not None:
            _append_llm_usage_journal(
                journal_path,
                {
                    "invocation_id": invocation_id,
                    "episode_index": episode_index + 1,
                    **episode_record,
                },
            )

    return record_usage


def _list_rate_of(record: dict[str, Any]) -> float | None:
    """Return published-rate cost when known.

    Use cost_usd when it already represents the list rate; a billed-only price
    without a published estimate remains unavailable.
    """
    if _is_number(record.get("cost_usd_list_rate")):
        return float(record["cost_usd_list_rate"])
    if record.get("cost_usd_source") in ("litellm_cost_map", "no_tokens_billed"):
        return record.get("cost_usd")
    return None


def summarize_llm_usage(
    records: list[dict[str, Any]],
    *,
    include_actor_breakdown: bool = True,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "call_count": len(records),
        "accepted_call_count": sum(
            record.get("response_status") == "accepted" for record in records
        ),
        "retry_response_count": sum(
            record.get("response_status") != "accepted" for record in records
        ),
    }
    for field in TOKEN_FIELDS:
        values = [record.get(field) for record in records]
        reported = [value for value in values if isinstance(value, int)]
        summary[field] = sum(reported)
        summary[f"{field}_reported_calls"] = len(reported)
        summary[f"{field}_missing_calls"] = len(values) - len(reported)

    reasoning_sources: dict[str, int] = {}
    for record in records:
        source = str(record.get("reasoning_tokens_source", "unavailable"))
        reasoning_sources[source] = reasoning_sources.get(source, 0) + 1
    summary["reasoning_tokens_sources"] = reasoning_sources

    # Track billed and published-rate totals separately.
    for field, price_of in (
        ("cost_usd", lambda record: record.get("cost_usd")),
        ("cost_usd_list_rate", _list_rate_of),
    ):
        prices = [price_of(record) for record in records]
        priced = [float(price) for price in prices if _is_number(price)]
        summary[field] = round(sum(priced), 6)
        summary[f"{field}_reported_calls"] = len(priced)
        summary[f"{field}_missing_calls"] = len(prices) - len(priced)

    if include_actor_breakdown:
        actors = sorted({str(record.get("actor", "unknown")) for record in records})
        summary["by_actor"] = {
            actor: summarize_llm_usage(
                [
                    record
                    for record in records
                    if str(record.get("actor", "unknown")) == actor
                ],
                include_actor_breakdown=False,
            )
            for actor in actors
        }
    return summary
