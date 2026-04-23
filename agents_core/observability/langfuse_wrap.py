"""Optional Langfuse observability for LLM calls.

Singleton: first call inits, reused afterwards. If langfuse pkg missing or
LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY env vars not set, every helper
returns None / no-ops so caller code runs unchanged.

Ported from unified-telegram-bot/app/services/_langfuse_wrap.py (2026-04-23).
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse  # type: ignore

    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False

# USD per 1M tokens — keep in sync with llm/client.py _PRICING.
_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "deepseek-chat": {"input": 0.27, "output": 1.10},
}

_langfuse: Any | None = None
_init_attempted = False


def get_langfuse() -> Any | None:
    """Return singleton Langfuse client, or None if disabled/missing credentials."""
    global _langfuse, _init_attempted
    if _init_attempted:
        return _langfuse
    _init_attempted = True

    if not _LANGFUSE_AVAILABLE:
        return None
    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        return None

    try:
        _langfuse = Langfuse()
        logger.info("[agents_core.observability] Langfuse enabled")
    except Exception as e:
        logger.warning(
            "[agents_core.observability] Langfuse init failed (%s: %s)",
            type(e).__name__, e,
        )
        _langfuse = None
    return _langfuse


def _reset_for_tests() -> None:
    global _langfuse, _init_attempted
    _langfuse = None
    _init_attempted = False


def calc_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Return cost in USD. Unknown models → 0.0."""
    pricing = _PRICING.get(model_id)
    if pricing is None:
        return 0.0
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def start_generation(
    name: str,
    model: str,
    user_message: str,
    system: str | None = None,
    max_tokens: int | None = None,
    extra_params: dict[str, Any] | None = None,
) -> Any | None:
    """Start a Langfuse generation span. Returns handle or None."""
    lf = get_langfuse()
    if lf is None:
        return None
    try:
        lf_input: list[dict] = []
        if system is not None:
            lf_input.append({"role": "system", "content": system})
        lf_input.append({"role": "user", "content": user_message})

        params: dict[str, Any] = {}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if extra_params:
            params.update(extra_params)

        return lf.start_observation(
            as_type="generation",
            name=name,
            model=model,
            model_parameters=params or None,
            input=lf_input,
        )
    except Exception as e:
        logger.debug("[agents_core.observability] start_observation failed: %s", e)
        return None


def end_generation_ok(
    gen: Any | None,
    output_text: Any,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    duration_sec: float,
    attempt: int = 1,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Close a generation span with success. No-op if gen is None."""
    if gen is None:
        return
    try:
        cost = calc_cost(model_id, input_tokens, output_tokens)
        usage: dict[str, int] = {"input": input_tokens, "output": output_tokens}
        if cache_creation_input_tokens:
            usage["cache_creation_input_tokens"] = cache_creation_input_tokens
        if cache_read_input_tokens:
            usage["cache_read_input_tokens"] = cache_read_input_tokens
        meta: dict[str, Any] = {"cost_usd": cost, "duration_sec": duration_sec, "attempt": attempt}
        if extra_metadata:
            meta.update(extra_metadata)
        gen.update(
            output=(
                {"role": "assistant", "content": output_text}
                if isinstance(output_text, str)
                else output_text
            ),
            usage_details=usage,
            metadata=meta,
        )
        gen.end()
    except Exception as e:
        logger.debug("[agents_core.observability] end (ok) failed: %s", e)


def end_generation_err(
    gen: Any | None,
    error: BaseException,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Close a generation span with error. No-op if gen is None."""
    if gen is None:
        return
    try:
        meta = dict(extra_metadata or {})
        gen.update(
            level="ERROR",
            status_message=f"{type(error).__name__}: {error}",
            metadata=meta or None,
        )
        gen.end()
    except Exception as e:
        logger.debug("[agents_core.observability] end (err) failed: %s", e)
