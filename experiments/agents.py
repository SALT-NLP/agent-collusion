"""LiteLLM calls, retry handling, tool execution, and usage normalization."""

import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

import litellm
from litellm import completion
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnsupportedParamsError,
)

from experiments.llm import (
    get_litellm_completion_kwargs,
    reasoning_to_replay,
    resolve_provider,
    with_prompt_cache_breakpoint,
    with_prose_split_from_tool_calls,
)
from experiments.memory.messages import model_visible_messages
from experiments.models import (
    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
)
from experiments.protocol.dispatch import (
    execute_tool_call,
    format_tool_content,
    reject_undecodable_arguments,
)
from experiments.protocol.state import agent_state, peer_id
from experiments.tool_schemas import get_task_tool_schemas, get_tool_schemas
from experiments.usage import _get_attr, _jsonable, normalize_llm_usage

LLM_MAX_RETRIES = 20
LLM_RETRY_BASE_WAIT_SECONDS = 5
LLM_RETRY_MAX_WAIT_SECONDS = 60
LLM_RETRY_MAX_TOTAL_WAIT_SECONDS = 20 * 60
# Bound provider error details to keep retry logs readable.
LLM_LOG_EXCERPT_CHARS = 400
litellm.suppress_debug_info = True
NON_RETRYABLE_LLM_ERRORS = (
    AuthenticationError,
    ContextWindowExceededError,
    NotFoundError,
    PermissionDeniedError,
    UnsupportedParamsError,
)
UsageRecorder = Callable[[dict[str, Any]], None]
CONTENT_FILTER_FINISH_REASONS = frozenset(
    {
        "blocklist",
        "content_filter",
        "image_prohibited_content",
        "image_safety",
        "jailbreak",
        "language",
        "model_armor",
        "prohibited_content",
        "recitation",
        "safety",
        "spii",
    }
)
# Include provider block reasons and safety ratings when available.
CONTENT_FILTER_DETAIL_FIELDS = (
    "prompt_feedback",
    "vertex_ai_safety_results",
    "vertex_ai_citation_metadata",
)


@dataclass(frozen=True)
class LLMErrorDecision:
    category: str
    retryable: bool
    retry_after_seconds: float | None = None


class LLMRetryLimitExceededError(RuntimeError):
    pass


def _extract_reasoning_trace(message: Any) -> dict[str, Any] | None:
    """Extract provider reasoning fields without adding them to agent memory."""
    raw = _jsonable(message)
    candidates: dict[str, Any] = {}

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key)
                next_path = f"{path}.{key_text}" if path else key_text
                lowered = key_text.lower()
                if "reason" in lowered or "thinking" in lowered or "thought" in lowered:
                    if item not in (None, "", [], {}):
                        candidates[next_path] = _jsonable(item)
                visit(item, next_path)
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                visit(item, f"{path}[{idx}]")

    visit(raw)
    if not candidates:
        return None
    return candidates


def _normalize_tool_calls(tool_calls_raw: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not tool_calls_raw:
        return normalized
    for tc in tool_calls_raw:
        function_obj = _get_attr(tc, "function", {})
        normalized_function = {
            "name": _get_attr(function_obj, "name", ""),
            "arguments": _get_attr(function_obj, "arguments", "{}"),
        }
        function_provider_fields = _get_attr(
            function_obj,
            "provider_specific_fields",
        )
        if function_provider_fields:
            normalized_function["provider_specific_fields"] = _jsonable(
                function_provider_fields
            )

        normalized_tool_call = {
            "id": _get_attr(tc, "id", ""),
            "type": _get_attr(tc, "type", "function"),
            "function": normalized_function,
        }
        for metadata_key in ("extra_content", "provider_specific_fields"):
            metadata = _get_attr(tc, metadata_key)
            if metadata:
                normalized_tool_call[metadata_key] = _jsonable(metadata)
        normalized.append(normalized_tool_call)
    return normalized


def _forced_tool_name(tool_choice: str | dict[str, Any]) -> str | None:
    if not isinstance(tool_choice, dict):
        return None
    name = tool_choice.get("name")
    if isinstance(name, str) and name:
        return name
    function = tool_choice.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name:
            return name
    return None


def _select_tools_and_choice(
    *,
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any],
    model: str,
    reasoning_effort: str,
) -> tuple[list[dict[str, Any]], str | dict[str, Any]]:
    forced_tool_name = _forced_tool_name(tool_choice)
    use_auto = _needs_auto_tool_choice_fallback(model, reasoning_effort)
    if forced_tool_name:
        matching_tools = [
            tool
            for tool in tools
            if _get_attr(_get_attr(tool, "function", {}), "name", "")
            == forced_tool_name
        ]
        if use_auto:
            return (matching_tools or tools), "auto"
        return tools, {
            "type": "function",
            "function": {"name": forced_tool_name},
        }
    if use_auto and tool_choice == "required":
        return tools, "auto"
    return tools, tool_choice


def _decoded_tool_arguments(raw_arguments: Any) -> dict[str, Any] | None:
    """Decode tool arguments from a JSON string or mapping; return None for non-objects."""
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not raw_arguments:
        return {}
    try:
        decoded = json.loads(raw_arguments)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _execute_model_tool_calls(
    *,
    state: dict[str, Any],
    actor: str,
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    executed: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        arguments = _decoded_tool_arguments(tool_call["function"]["arguments"])
        if arguments is None:
            arguments = {}
            result = reject_undecodable_arguments(
                state=state,
                actor=actor,
                tool_name=tool_name,
            )
        else:
            result = execute_tool_call(
                state=state,
                actor=actor,
                tool_name=tool_name,
                arguments=arguments,
            )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "content": format_tool_content(result),
            }
        )
        executed.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
            }
        )
    return executed


def _retry_wait_seconds(
    retry_idx: int,
    retry_after_seconds: float | None = None,
) -> float:
    backoff = min(
        LLM_RETRY_BASE_WAIT_SECONDS * (2**retry_idx),
        LLM_RETRY_MAX_WAIT_SECONDS,
    )
    wait = max(backoff, retry_after_seconds or 0.0)
    # Jitter prevents concurrent runs from retrying a rate-limited route together.
    return wait + random.uniform(0.0, min(1.0, backoff * 0.1))


def _error_evidence(exc: Exception) -> str:
    """All provider error material retained by LiteLLM, as searchable text."""
    evidence: list[str] = [str(exc)]
    for attribute in ("body", "detail", "provider_specific_fields"):
        value = _get_attr(exc, attribute)
        if value not in (None, "", {}, []):
            evidence.append(json.dumps(_jsonable(value), ensure_ascii=False))

    response = _get_attr(exc, "response")
    if response is not None:
        try:
            body = response.json()
        except Exception:
            body = _get_attr(response, "text", "")
        if body not in (None, "", {}, []):
            evidence.append(
                body
                if isinstance(body, str)
                else json.dumps(_jsonable(body), ensure_ascii=False)
            )
    return "\n".join(evidence)


def _retry_after_header_seconds(value: str) -> float | None:
    value = value.strip()
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0)


def _header_pairs(source: Any) -> list[tuple[str, str]]:
    """Read exception headers from mappings or SDK objects exposing ``items()``."""
    items = getattr(source, "items", None)
    if callable(items):
        try:
            candidates: Any = list(items())
        except Exception:
            return []
    elif isinstance(source, (list, tuple)):
        candidates = source
    else:
        return []
    return [
        (str(pair[0]), str(pair[1]))
        for pair in candidates
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ]


def _provider_retry_after_seconds(exc: Exception, evidence: str) -> float | None:
    """Read HTTP Retry-After and google.rpc RetryInfo from a provider error."""
    header_values: list[str] = []
    for source in (
        _get_attr(exc, "headers"),
        _get_attr(_get_attr(exc, "response"), "headers"),
    ):
        if not source:
            continue
        for key, value in _header_pairs(source):
            if key.lower() == "retry-after":
                header_values.append(value)

    candidates = [
        seconds
        for value in header_values
        if (seconds := _retry_after_header_seconds(value)) is not None
    ]
    duration_patterns = (
        r'["\']retryDelay["\']\s*:\s*["\']([0-9]+(?:\.[0-9]+)?)s["\']',
        r'["\']retry_delay["\']\s*:\s*["\']([0-9]+(?:\.[0-9]+)?)s["\']',
        r"\bretry\s+in\s+([0-9]+(?:\.[0-9]+)?)s\b",
    )
    for pattern in duration_patterns:
        candidates.extend(
            float(match) for match in re.findall(pattern, evidence, re.IGNORECASE)
        )
    return max(candidates) if candidates else None


def _is_semantic_rate_limit(exc: Exception, evidence: str) -> bool:
    """Whether the provider actually reported 429, despite wrapper taxonomy."""
    if isinstance(exc, RateLimitError):
        return True
    response_status = _get_attr(_get_attr(exc, "response"), "status_code")
    if response_status == 429 or _get_attr(exc, "status_code") == 429:
        return True
    return bool(
        re.search(r'["\']?code["\']?\s*:\s*429\b', evidence, re.IGNORECASE)
        or re.search(
            r'["\']?status["\']?\s*:\s*["\']RESOURCE_EXHAUSTED["\']',
            evidence,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:rate_limit_exceeded|quota_exceeded|too many requests)\b",
            evidence,
            re.IGNORECASE,
        )
    )


def _is_hard_quota(exc: Exception, evidence: str) -> bool:
    """A quota that an identical short-delay retry cannot make available."""
    rate_limit_type = str(_get_attr(exc, "rate_limit_type", "")).lower()
    if rate_limit_type in {"budget", "max_iterations"}:
        return True
    hard_patterns = (
        r"\b(?:insufficient_quota|quota_exceeded)\b",
        r"\b(?:requests?|tokens?|images?)\s+per\s+day\b",
        r"\b(?:rpd|tpd|ipd)\b",
        r"\bdaily\s+(?:quota|limit)\b",
        r"per[_ -]?day(?:\b|per)",
        r"\b(?:quota|rate)[_-]?limit[^\n]{0,40}\bper[_ -]?day\b",
        r"\b(?:billing|payment)\s+(?:account\s+)?(?:is\s+)?"
        r"(?:disabled|inactive|not active|not linked|required)\b",
        # Match zero exactly, excluding fractional limits such as 0.5.
        r'["\'](?:quotaValue|quota_value|limit)["\']\s*:\s*'
        r'["\']?0(?:\.0+)?["\']?(?![0-9.])',
    )
    return any(re.search(pattern, evidence, re.IGNORECASE) for pattern in hard_patterns)


def _classify_llm_error(exc: Exception) -> LLMErrorDecision:
    """Classify provider errors using their payload, not just the wrapper exception.

    Short-term rate limits may recover; daily or zero quotas do not.
    """
    evidence = _error_evidence(exc)
    if isinstance(exc, ContentPolicyViolationError):
        # Input rejection is not retryable; response filtering is handled after generation.
        return LLMErrorDecision("content_policy", False)

    if _is_semantic_rate_limit(exc, evidence):
        retry_after = _provider_retry_after_seconds(exc, evidence)
        if _is_hard_quota(exc, evidence):
            return LLMErrorDecision("hard_quota", False, retry_after)
        return LLMErrorDecision("transient_rate_limit", True, retry_after)

    if isinstance(exc, (BadRequestError, *NON_RETRYABLE_LLM_ERRORS)):
        return LLMErrorDecision("invalid_request", False)
    return LLMErrorDecision(
        "transient_llm_error",
        True,
        _provider_retry_after_seconds(exc, evidence),
    )


def _content_filter_reason(finish_reason: Any) -> str | None:
    normalized = str(finish_reason or "").strip().lower()
    return normalized if normalized in CONTENT_FILTER_FINISH_REASONS else None


def _log_excerpt(text: str, limit: int = LLM_LOG_EXCERPT_CHARS) -> str:
    """Unescape and flatten nested provider errors into one readable log line."""
    flat = text.replace("\\n", " ").replace('\\"', '"').replace("\\/", "/")
    flat = re.sub(r"\s+", " ", flat).strip()
    return flat if len(flat) <= limit else flat[:limit].rstrip() + " …"


def _content_filter_detail(response: Any, choice: Any) -> str:
    """Extract optional provider block details beyond the finish reason."""
    details: list[str] = []
    for field in CONTENT_FILTER_DETAIL_FIELDS:
        value = _get_attr(response, field)
        if value not in (None, "", {}, []):
            details.append(
                f"{field}={json.dumps(_jsonable(value), ensure_ascii=False)}"
            )
    for holder in (choice, _get_attr(choice, "message")):
        value = _get_attr(holder, "provider_specific_fields")
        if value not in (None, "", {}, []):
            details.append(json.dumps(_jsonable(value), ensure_ascii=False))
    return _log_excerpt(" ".join(details)) if details else ""


def _response_status_for_error(decision: LLMErrorDecision) -> str:
    return {
        "content_policy": "error_content_policy",
        "hard_quota": "error_quota_non_retryable",
        "invalid_request": "error_non_retryable",
    }.get(decision.category, "error_retryable")


def _record_classified_error(
    *,
    exc: Exception,
    model: str,
    retry_index: int,
    decision: LLMErrorDecision,
    usage_recorder: UsageRecorder | None,
) -> None:
    record = _error_usage_record(
        exc,
        model,
        retry_index + 1,
        _response_status_for_error(decision),
    )
    record.update(
        {
            "error_category": decision.category,
            "retry_after_seconds": decision.retry_after_seconds,
        }
    )
    _record_usage(record, usage_recorder)


def _format_wait(wait: float) -> str:
    return f"{wait:.3f}".rstrip("0").rstrip(".")


def _raise_if_retry_exhausted(
    retry_idx: int,
    reason: str,
    *,
    waited_seconds: float = 0.0,
    next_wait_seconds: float = 0.0,
) -> None:
    if retry_idx >= LLM_MAX_RETRIES:
        raise LLMRetryLimitExceededError(
            f"LLM retry limit exceeded after {retry_idx + 1} API attempts "
            f"({LLM_MAX_RETRIES} retries): {reason}"
        )
    if waited_seconds + next_wait_seconds > LLM_RETRY_MAX_TOTAL_WAIT_SECONDS:
        raise LLMRetryLimitExceededError(
            "LLM retry wait limit exceeded before the next API attempt "
            f"({LLM_RETRY_MAX_TOTAL_WAIT_SECONDS}s maximum; provider/backoff "
            f"requested {_format_wait(next_wait_seconds)}s): {reason}"
        )


def _record_usage(
    usage_record: dict[str, Any],
    usage_recorder: UsageRecorder | None,
) -> None:
    """Send call usage to the recorder for episode accounting and journal persistence."""
    if usage_recorder is not None:
        usage_recorder(_jsonable(usage_record))


def _error_usage_record(
    exc: Exception,
    requested_model: str,
    api_attempt: int,
    response_status: str,
) -> dict[str, Any]:
    usage_record = normalize_llm_usage(
        _get_attr(exc, "response"),
        requested_model,
        call_failed=True,
    )
    usage_record.update(
        {
            "api_attempt": api_attempt,
            "response_status": response_status,
            "finish_reason": None,
            "choice_count": None,
            "visible_content_chars": None,
            "tool_call_count": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    )
    return usage_record


def _needs_auto_tool_choice_fallback(
    model: str,
    reasoning_effort: str,
) -> bool:
    """Identify routes that reject forced tool choices.

    Thinking DeepSeek requests require an unforced choice, including at the default
    effort.
    """
    reasoning_effort = reasoning_effort.strip().lower()
    return resolve_provider(model)[1] == "deepseek" and reasoning_effort != "none"


def _complete_with_retries(
    *,
    model: str,
    messages: list[dict[str, Any]],
    completion_kwargs: dict[str, Any],
    temperature: float,
    max_output_tokens: int,
    accept_tool_calls: bool,
    usage_recorder: UsageRecorder | None,
) -> tuple[Any, Any, str, list[dict[str, Any]]]:
    retry_index = 0
    retry_waited_seconds = 0.0
    while True:
        try:
            # Apply provider-specific history transformations only to the outgoing request.
            response = completion(
                model=model,
                messages=with_prompt_cache_breakpoint(
                    model,
                    with_prose_split_from_tool_calls(
                        model,
                        model_visible_messages(messages),
                        completion_kwargs,
                    ),
                ),
                temperature=temperature,
                # LiteLLM's provider-neutral name for the output-token ceiling.
                max_tokens=max_output_tokens,
                **completion_kwargs,
            )
            usage_record = normalize_llm_usage(response, model)
            usage_record["api_attempt"] = retry_index + 1
            choices = _get_attr(response, "choices", []) or []
            usage_record["choice_count"] = len(choices)
            finish_reason = (
                _jsonable(_get_attr(choices[0], "finish_reason")) if choices else None
            )
            usage_record["finish_reason"] = finish_reason
            if not choices:
                usage_record.update(
                    {
                        "response_status": "empty_no_choices",
                        "visible_content_chars": 0,
                        "tool_call_count": 0,
                    }
                )
                _record_usage(usage_record, usage_recorder)
                wait = _retry_wait_seconds(retry_index)
                _raise_if_retry_exhausted(
                    retry_index,
                    "empty response with no choices",
                    waited_seconds=retry_waited_seconds,
                    next_wait_seconds=wait,
                )
                print(
                    f"    [empty response retry {retry_index + 1}/"
                    f"{LLM_MAX_RETRIES} in {_format_wait(wait)}s: no choices]",
                    flush=True,
                )
                time.sleep(wait)
                retry_waited_seconds += wait
                retry_index += 1
                continue

            choice_message = _get_attr(choices[0], "message")
            content = _get_attr(choice_message, "content", "") or ""
            tool_calls = (
                _normalize_tool_calls(_get_attr(choice_message, "tool_calls", []))
                if accept_tool_calls
                else []
            )
            usage_record["visible_content_chars"] = len(content)
            usage_record["tool_call_count"] = len(tool_calls)
            if content.strip() or tool_calls:
                usage_record["response_status"] = "accepted"
                _record_usage(usage_record, usage_recorder)
                return response, choice_message, content, tool_calls

            # Keep partial content from filtered responses; retry only empty filtered responses.
            content_filter_reason = _content_filter_reason(finish_reason)
            if content_filter_reason is not None:
                error_message = (
                    "LLM response blocked by content policy: "
                    f"{content_filter_reason}"
                )
                # Keep full block details separate from the retry-limit error label.
                detail = _content_filter_detail(response, choices[0])
                usage_record.update(
                    {
                        "response_status": "error_content_policy",
                        "error_category": "content_policy",
                        "error_type": "ContentPolicyBlockedResponse",
                        "error_message": error_message,
                    }
                )
                if detail:
                    usage_record["error_detail"] = detail
                _record_usage(usage_record, usage_recorder)
                wait = _retry_wait_seconds(retry_index)
                _raise_if_retry_exhausted(
                    retry_index,
                    f"{error_message}; {detail}" if detail else error_message,
                    waited_seconds=retry_waited_seconds,
                    next_wait_seconds=wait,
                )
                print(
                    f"    [content policy retry {retry_index + 1}/"
                    f"{LLM_MAX_RETRIES} in {_format_wait(wait)}s: "
                    f"{content_filter_reason}"
                    f"{f'; {detail}' if detail else ''}]",
                    flush=True,
                )
                time.sleep(wait)
                retry_waited_seconds += wait
                retry_index += 1
                continue

            status = (
                "empty_no_content_or_tool_calls"
                if accept_tool_calls
                else "empty_no_content"
            )
            reason = (
                "empty response without content or tool calls"
                if accept_tool_calls
                else "empty response without content"
            )
            usage_record["response_status"] = status
            _record_usage(usage_record, usage_recorder)
            wait = _retry_wait_seconds(retry_index)
            _raise_if_retry_exhausted(
                retry_index,
                reason,
                waited_seconds=retry_waited_seconds,
                next_wait_seconds=wait,
            )
            print(
                f"    [empty response retry {retry_index + 1}/"
                f"{LLM_MAX_RETRIES} in {_format_wait(wait)}s]",
                flush=True,
            )
            time.sleep(wait)
            retry_waited_seconds += wait
            retry_index += 1
        except LLMRetryLimitExceededError:
            raise
        except Exception as exc:
            decision = _classify_llm_error(exc)
            _record_classified_error(
                exc=exc,
                model=model,
                retry_index=retry_index,
                decision=decision,
                usage_recorder=usage_recorder,
            )
            if not decision.retryable:
                # Log the classification and provider detail before raising the transport error.
                print(
                    f"    [LLM error, not retried "
                    f"({decision.category}): {_log_excerpt(str(exc))}]",
                    flush=True,
                )
                raise
            wait = _retry_wait_seconds(
                retry_index,
                decision.retry_after_seconds,
            )
            _raise_if_retry_exhausted(
                retry_index,
                str(exc),
                waited_seconds=retry_waited_seconds,
                next_wait_seconds=wait,
            )
            print(
                f"    [LLM error, retry {retry_index + 1}/"
                f"{LLM_MAX_RETRIES} in {_format_wait(wait)}s "
                f"({decision.category}): {_log_excerpt(str(exc))}]",
                flush=True,
            )
            time.sleep(wait)
            retry_waited_seconds += wait
            retry_index += 1


def run_agent_turn(
    actor: str,
    model: str,
    messages: list[dict[str, Any]],
    state: dict[str, Any],
    tool_choice: str | dict[str, Any] = "required",
    reasoning_effort: str = "default",
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    usage_recorder: UsageRecorder | None = None,
) -> dict[str, Any]:
    # answer_key belongs to the agent task; task_type and phase are shared.
    slot = agent_state(state, actor)
    tools = get_tool_schemas(
        task_type=str(state.get("task_type", "code_analysis")),
        phase=str(state.get("phase", "task")),
        answer_key=[str(column) for column in slot.get("answer_key", [])],
        peer=peer_id(actor),
    )
    extra = get_litellm_completion_kwargs(
        model,
        reasoning_effort,
        passthrough_params=("tool_choice",),
    )
    tools, effective_tool_choice = _select_tools_and_choice(
        tools=tools,
        tool_choice=tool_choice,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    response, choice_message, assistant_content, tool_calls = _complete_with_retries(
        model=model,
        messages=messages,
        completion_kwargs={
            "tools": tools,
            "tool_choice": effective_tool_choice,
            **extra,
        },
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        accept_tool_calls=True,
        usage_recorder=usage_recorder,
    )

    reasoning_trace = _extract_reasoning_trace(choice_message)

    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_content,
    }
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    # Preserve provider reasoning on prose turns as well as tool-call turns.
    assistant_msg.update(reasoning_to_replay(model, choice_message))
    messages.append(assistant_msg)

    executed_tool_calls = _execute_model_tool_calls(
        state=state,
        actor=actor,
        messages=messages,
        tool_calls=tool_calls,
    )

    return {
        "tool_calls": executed_tool_calls,
        "raw_response_model": _get_attr(response, "model", model),
        "reasoning_trace": reasoning_trace,
        "reasoning_trace_chars": len(json.dumps(reasoning_trace, ensure_ascii=False))
        if reasoning_trace
        else 0,
    }


def run_reflection_turn(
    model: str,
    messages: list[dict[str, Any]],
    state: dict[str, Any],
    actor: str,
    reasoning_effort: str = "default",
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    usage_recorder: UsageRecorder | None = None,
) -> dict[str, Any]:
    """Generate a prose reflection from the agent history without executing tools.

    Include tool schemas because Anthropic/Bedrock require them for tool-use history.
    No tool choice is forced, and any returned tool calls are discarded.
    """
    slot = agent_state(state, actor)
    tools = get_task_tool_schemas(
        task_type=str(state.get("task_type", "code_analysis")),
        answer_key=[str(column) for column in slot.get("answer_key", [])],
        peer=peer_id(actor),
    )
    extra = get_litellm_completion_kwargs(model, reasoning_effort)
    response, choice_message, content, _ = _complete_with_retries(
        model=model,
        messages=messages,
        completion_kwargs={"tools": tools, **extra},
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        accept_tool_calls=False,
        usage_recorder=usage_recorder,
    )

    reasoning_trace = _extract_reasoning_trace(choice_message)
    replayed_reasoning = reasoning_to_replay(model, choice_message)
    messages.append({"role": "assistant", "content": content, **replayed_reasoning})
    return {
        "content": content,
        # Return reasoning for the caller to persist; this reflection used a history copy.
        "replayed_reasoning": replayed_reasoning,
        "raw_response_model": _get_attr(response, "model", model),
        "reasoning_trace": reasoning_trace,
        "reasoning_trace_chars": len(json.dumps(reasoning_trace, ensure_ascii=False))
        if reasoning_trace
        else 0,
    }
