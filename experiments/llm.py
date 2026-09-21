"""Provider routing, request adaptation, and reasoning replay through LiteLLM.

Model prefixes select providers; LiteLLM reads their credentials from the environment.
This module adds route checks, metadata, cache breakpoints, and provider-specific
history handling without changing the stored conversation.
"""

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Iterable

import litellm

from experiments.models import (
    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
)

# Reasoning efforts in ascending order after the special values.
# Supported levels and provider mappings vary by model.
REASONING_EFFORT_CHOICES = (
    "default",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)

# Treat transient provider failures as reachable during pre-flight checks.
REACHABLE_DESPITE_ERRORS = (
    litellm.RateLimitError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.Timeout,
)


def resolve_provider(model: str) -> tuple[str, str]:
    """Split a LiteLLM model string the way LiteLLM itself will read it."""
    try:
        model_name, provider, _, _ = litellm.get_llm_provider(model=model)
    except Exception:
        prefix, _, remainder = model.partition("/")
        return (remainder or prefix), (prefix if remainder else "openai")
    return model_name, provider


def effective_api_base(model: str) -> str:
    """Resolve an environment-configured base URL, or return an empty string.

    Check ``<PROVIDER>_BASE_URL`` before ``<PROVIDER>_API_BASE``.
    """
    provider = resolve_provider(model)[1].upper()
    for variable in (f"{provider}_BASE_URL", f"{provider}_API_BASE"):
        value = os.environ.get(variable, "").strip()
        if value:
            return value.rstrip("/")
    return ""


def _params_the_provider_cannot_map(
    model_name: str,
    provider: str,
    params: Iterable[str],
) -> list[str]:
    """Return parameters LiteLLM cannot map for this model.

    ``allowed_openai_params`` forwards fields verbatim. Override only unmapped fields
    to avoid sending raw OpenAI parameters alongside provider-specific equivalents.
    """
    requested = list(params)
    if not requested:
        return []
    try:
        supported = (
            litellm.get_supported_openai_params(
                model=model_name,
                custom_llm_provider=provider,
            )
            or []
        )
    except Exception:
        supported = []
    return [param for param in requested if param not in supported]


# Explicit budgets corresponding to LiteLLM's standard low/medium/high levels.
_THINKING_BUDGET_BY_EFFORT = {"low": 1024, "medium": 2048, "high": 4096}


def _thinking_instead_of_effort(
    model_name: str,
    provider: str,
    reasoning_effort: str,
) -> dict[str, Any] | None:
    """Use an explicit thinking budget for adaptive Anthropic models when supported.

    LiteLLM's Bedrock path converts forced tool choices to auto for enabled thinking,
    but not for adaptive thinking. Explicit budgets preserve that conversion.
    If the private model probe is unavailable, leave the requested effort unchanged.
    """
    budget = _THINKING_BUDGET_BY_EFFORT.get(reasoning_effort)
    if budget is None:
        return None
    try:
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        adaptive = AnthropicConfig._is_adaptive_thinking_model(model_name, provider)
    except Exception:
        return None
    if not adaptive:
        return None
    return {"type": "enabled", "budget_tokens": budget}


def _deepseek_effort_kwargs(reasoning_effort: str) -> dict[str, Any]:
    """Preserve DeepSeek's reasoning level or explicit thinking-disable switch.

    Force these fields through ``allowed_openai_params`` because LiteLLM's mapping
    otherwise drops the level or the disable switch.
    """
    if reasoning_effort == "none":
        return {"thinking": {"type": "disabled"}}
    return {"reasoning_effort": reasoning_effort}


def get_litellm_completion_kwargs(
    model: str,
    reasoning_effort: str,
    passthrough_params: Iterable[str] = (),
) -> dict[str, Any]:
    """Build reasoning kwargs and passthrough overrides for ``litellm.completion``.

    Credentials come from the environment. ``passthrough_params`` lists additional
    OpenAI fields that must reach the endpoint, such as ``tool_choice``.
    """
    model_name, provider = resolve_provider(model)
    kwargs: dict[str, Any] = {}
    requested_params = list(passthrough_params)
    forced_params: list[str] = []
    normalized_reasoning_effort = reasoning_effort.strip().lower()
    if normalized_reasoning_effort and normalized_reasoning_effort != "default":
        if provider == "deepseek":
            effort_kwargs = _deepseek_effort_kwargs(normalized_reasoning_effort)
            kwargs.update(effort_kwargs)
            # DeepSeek requires overrides even for fields LiteLLM reports as supported.
            forced_params.extend(effort_kwargs)
        else:
            thinking = _thinking_instead_of_effort(
                model_name,
                provider,
                normalized_reasoning_effort,
            )
            if thinking is None:
                kwargs["reasoning_effort"] = normalized_reasoning_effort
                requested_params.append("reasoning_effort")
            else:
                kwargs["thinking"] = thinking
                requested_params.append("thinking")

    forced_params.extend(
        _params_the_provider_cannot_map(
            model_name,
            provider,
            requested_params,
        )
    )
    if forced_params:
        kwargs["allowed_openai_params"] = forced_params
    return kwargs


def _speaks_anthropic_wire_format(model_name: str, provider: str) -> bool:
    """Check the provider wire format before replaying Anthropic thinking blocks.

    A Claude deployment behind an OpenAI-compatible proxy does not use this format.
    """
    if provider == "anthropic":
        return True
    return provider == "bedrock" and "anthropic." in model_name.lower()


# Bedrock Converse accepts message-level cache breakpoints only on these roles.
_CACHE_BREAKPOINT_ROLES = frozenset({"system", "user", "tool"})


def marks_prompt_cache_breakpoint(model: str) -> bool:
    """Check whether the route requires an explicit Anthropic cache breakpoint.

    Generic prompt-caching support does not imply support for breakpoint fields.
    """
    return _speaks_anthropic_wire_format(*resolve_provider(model))


def with_prompt_cache_breakpoint(
    model: str,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add one cache breakpoint at the newest cacheable turn.

    Copy the affected message so breakpoints do not accumulate in stored history.
    """
    if not messages or not marks_prompt_cache_breakpoint(model):
        return messages
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") in _CACHE_BREAKPOINT_ROLES:
            marked = dict(messages[index])
            marked["cache_control"] = {"type": "ephemeral"}
            return [*messages[:index], marked, *messages[index + 1 :]]
    return messages


def bridges_to_responses_api(model: str, completion_kwargs: dict[str, Any]) -> bool:
    """Ask LiteLLM whether these request kwargs trigger the Responses API bridge.

    Return False if its private bridge check is unavailable.
    """
    try:
        from litellm.main import responses_api_bridge_check

        model_name, provider = resolve_provider(model)
        model_info, _ = responses_api_bridge_check(
            model=model_name,
            custom_llm_provider=provider,
            tools=completion_kwargs.get("tools"),
            reasoning_effort=completion_kwargs.get("reasoning_effort"),
        )
    except Exception:
        return False
    return model_info.get("mode") == "responses"


def with_prose_split_from_tool_calls(
    model: str,
    messages: list[dict[str, Any]],
    completion_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Split assistant prose from tool calls for requests using the Responses bridge.

    The bridge omits text on tool-call messages. Split only the outgoing history,
    keeping reasoning with prose and calls adjacent to their tool results.
    """
    if not messages or not bridges_to_responses_api(model, completion_kwargs):
        return messages
    sent: list[dict[str, Any]] = []
    for message in messages:
        spoke = str(message.get("content") or "").strip()
        if (
            message.get("role") != "assistant"
            or not spoke
            or not message.get("tool_calls")
        ):
            sent.append(message)
            continue
        sent.append(
            {key: value for key, value in message.items() if key != "tool_calls"}
        )
        sent.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": message["tool_calls"],
            }
        )
    return sent


# Provider-specific fields used to replay prior reasoning:
#   reasoning_content: DeepSeek; Qwen also needs reasoning for vLLM validation.
#   thinking_blocks: Anthropic signed reasoning blocks.
#   reasoning_items: OpenAI encrypted items through the Responses API bridge.
#   thought_signatures: Gemini signatures attached to their original parts.


def _response_field(message: Any, name: str) -> Any:
    """Read a response field from either a mapping or a LiteLLM object."""
    value = getattr(message, name, None)
    if value is None and isinstance(message, dict):
        value = message.get(name)
    return value


def _copied_field(
    source: str,
    targets: tuple[str, ...],
) -> Callable[[Any], dict[str, Any]]:
    """Build a replay extractor that copies a response field to the given target names."""

    def replay(message: Any) -> dict[str, Any]:
        reasoning = _response_field(message, source)
        if not reasoning:
            return {}
        return {target: reasoning for target in targets}

    return replay


def _tool_call_signature_carriers(message: Any) -> list[str]:
    """Collect tool-call strings that may already carry a Gemini thought signature.

    LiteLLM can encode signatures in provider fields or tool-call IDs.
    """
    carriers: list[str] = []
    for call in _response_field(message, "tool_calls") or []:
        carriers.append(str(_response_field(call, "id") or ""))
        for holder in (call, _response_field(call, "function")):
            fields = _response_field(holder, "provider_specific_fields")
            if isinstance(fields, dict):
                carriers.append(str(fields.get("thought_signature") or ""))
    return [carrier for carrier in carriers if carrier]


def _thought_part_signatures(message: Any) -> set[str]:
    """Collect signatures attached to thought parts from their thinking blocks."""
    return {
        str(block.get("signature"))
        for block in _response_field(message, "thinking_blocks") or []
        if isinstance(block, dict) and block.get("signature")
    }


def _gemini_thought_signature(message: Any) -> dict[str, Any]:
    """Select a Gemini signature for replay on the turn's text part.

    Use the last signature not attached to a tool call or thought block, and only
    when the turn has text. Tool-call signatures are replayed with their calls;
    thought signatures must not be attached to visible prose.
    """
    if not str(_response_field(message, "content") or "").strip():
        return {}
    fields = _response_field(message, "provider_specific_fields")
    signatures = [
        str(signature)
        for signature in (
            (fields if isinstance(fields, dict) else {}).get("thought_signatures") or []
        )
        if signature
    ]
    carriers = _tool_call_signature_carriers(message)
    on_a_thought = _thought_part_signatures(message)
    unplaced = [
        signature
        for signature in signatures
        if signature not in on_a_thought
        and not any(signature in carrier for carrier in carriers)
    ]
    if not unplaced:
        return {}
    return {"provider_specific_fields": {"thought_signatures": [unplaced[-1]]}}


_REASONING_DIALECTS: dict[str, Callable[[Any], dict[str, Any]]] = {
    # dialect -> what to keep on the assistant message, read off the response
    "deepseek": _copied_field("reasoning_content", ("reasoning_content",)),
    "anthropic": _copied_field("thinking_blocks", ("thinking_blocks",)),
    "qwen": _copied_field("reasoning_content", ("reasoning_content", "reasoning")),
    "openai": _copied_field("reasoning_items", ("reasoning_items",)),
    "gemini": _gemini_thought_signature,
}


def _reasoning_dialect(model: str) -> str:
    """Select the route's reasoning-replay dialect.

    Recognize DeepSeek and Qwen deployment names behind OpenAI-compatible proxies.
    Use provider format for Anthropic, and provider plus model for Vertex Gemini.
    Other OpenAI routes replay reasoning_items only when returned by the response.
    """
    model_name, provider = resolve_provider(model)
    deployment = model_name.lower()
    if provider == "deepseek" or "deepseek" in deployment:
        return "deepseek"
    if _speaks_anthropic_wire_format(model_name, provider):
        return "anthropic"
    if provider == "gemini" or (provider == "vertex_ai" and "gemini" in deployment):
        return "gemini"
    if "qwen" in deployment:
        return "qwen"
    if provider in ("openai", "azure"):
        return "openai"
    return ""


def reasoning_to_replay(model: str, message: Any) -> dict[str, Any]:
    """Extract replayable reasoning for any assistant turn, including prose-only turns.

    Return an empty mapping when no supported reasoning field is present.
    """
    dialect = _REASONING_DIALECTS.get(_reasoning_dialect(model))
    if dialect is None:
        return {}
    return dialect(message)


def get_litellm_version() -> str | None:
    """The installed client version that determines provider request mapping."""
    try:
        return version("litellm")
    except PackageNotFoundError:
        return None


def _missing_key_hint(model: str) -> str:
    """Describe expected provider credentials for diagnostics after a failed route check.

    This advisory list may omit valid authentication methods and does not gate calls.
    """
    try:
        audit = litellm.utils.validate_environment(model=model)
    except Exception:
        return ""
    if audit.get("keys_in_environment"):
        return ""
    missing = [str(name) for name in audit.get("missing_keys") or []]
    if not missing:
        return ""
    return f" LiteLLM expects one of: {', '.join(missing)}."


def validate_model_route(
    model: str,
    *,
    reasoning_effort: str = "default",
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    probe: bool = True,
) -> None:
    """Probe a model with a small request before starting the run."""
    if not probe:
        return
    try:
        # Build a fresh request because LiteLLM may mutate it.
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            temperature=temperature,
            # LiteLLM maps this output ceiling to max_completion_tokens,
            # max_output_tokens, or maxTokens for the selected provider.
            max_tokens=max_output_tokens,
            **get_litellm_completion_kwargs(model, reasoning_effort),
        )
    except REACHABLE_DESPITE_ERRORS:
        return
    except Exception as exc:
        raise ValueError(
            f"Model {model!r} could not be called: "
            f"{type(exc).__name__}: {exc}.{_missing_key_hint(model)}"
        ) from exc


def get_endpoint_metadata(model: str) -> dict[str, str]:
    """Return provider, endpoint, and client settings without credentials."""
    model_name, provider = resolve_provider(model)
    metadata = {"provider": provider, "model_name": model_name}
    api_base = effective_api_base(model)
    if api_base:
        metadata["api_base"] = api_base
    return metadata
