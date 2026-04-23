"""Unified async LLM client — Anthropic + DeepSeek with retries, caching, instructor, Langfuse.

Consolidates three legacy modules:
- unified-telegram-bot/app/services/_langfuse_wrap.py (observability wrapper)
- unified-telegram-bot/app/services/llm_router.py (Anthropic + DeepSeek routing)
- speech-analytics/src/llm_client.py (retries, prompt caching, instructor structured output)

Usage:
    client = LLMClient(anthropic_api_key=..., deepseek_api_key=None)
    resp: LLMResponse = await client.chat(
        prompt="...", model="haiku", system="...", system_cache=True,
        max_tokens=2048, name="myagent.step1",
    )
    # resp.text / resp.usage.{input,output,cache_creation,cache_read} / resp.cost_usd

    obj, raw = await client.chat_structured(
        prompt="...", response_model=MyModel, model="sonnet",
        system="...", name="myagent.analyze",
    )

Everything optional: no LANGFUSE_* env → spans are no-op; no deepseek_api_key → model
must be Anthropic; no instructor pkg → chat_structured raises clear error.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
from anthropic import AsyncAnthropic

from agents_core.observability import langfuse_wrap as lf

try:
    import instructor  # type: ignore
    from pydantic import BaseModel  # type: ignore

    _INSTRUCTOR_AVAILABLE = True
except ImportError:
    _INSTRUCTOR_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseModel")

MODEL_MAP: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
    "deepseek": "deepseek-chat",
    "deepseek-chat": "deepseek-chat",
}

_DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 2  # 1s, 2s, 4s


@dataclass
class LLMUsage:
    input: int = 0
    output: int = 0
    cache_creation: int = 0
    cache_read: int = 0


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    cost_usd: float = 0.0
    duration_sec: float = 0.0
    attempt: int = 1
    raw: Any = None  # underlying SDK response for callers that need it


@dataclass
class ToolUse:
    """A single tool_use block from an Anthropic agent turn."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class AgentTurn:
    """One ReAct turn: model + tools -> content blocks + stop_reason.

    `tool_uses` is the list of tool_use blocks the model emitted. `text` is
    the concatenated text from text blocks (if any). `message_content` is the
    raw Anthropic content list that MUST be echoed back as assistant message
    for the next turn (per Messages API contract).
    """

    model: str
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | "stop_sequence"
    text: str
    tool_uses: list[ToolUse]
    message_content: list[Any]  # raw content blocks (SDK objects or dicts)
    usage: LLMUsage = field(default_factory=LLMUsage)
    cost_usd: float = 0.0
    duration_sec: float = 0.0
    attempt: int = 1
    raw: Any = None


class LLMClient:
    """Async LLM client with unified interface across Anthropic and DeepSeek."""

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        deepseek_api_key: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_base: int = _DEFAULT_BACKOFF_BASE,
        temperature: float = 0.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._anthropic_key = anthropic_api_key
        self._deepseek_key = deepseek_api_key
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._temperature = temperature

        self._anthropic: AsyncAnthropic | None = (
            AsyncAnthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        )
        self._instructor_client: Any | None = None  # lazy
        self._http = http_client or httpx.AsyncClient(timeout=60.0)
        self._owns_http = http_client is None

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ chat
    async def chat(
        self,
        prompt: str,
        model: str = "haiku",
        system: str | None = None,
        system_cache: bool = False,
        max_tokens: int = 2048,
        name: str = "llm.chat",
    ) -> LLMResponse:
        """Text chat with retries + Langfuse span. Routes by model id."""
        model_id = MODEL_MAP.get(model, model)
        if model_id == "deepseek-chat":
            return await self._chat_deepseek(prompt, system, max_tokens, name)
        return await self._chat_anthropic(prompt, model_id, system, system_cache, max_tokens, name)

    async def _chat_anthropic(
        self,
        prompt: str,
        model_id: str,
        system: str | None,
        system_cache: bool,
        max_tokens: int,
        name: str,
    ) -> LLMResponse:
        if self._anthropic is None:
            raise RuntimeError("anthropic_api_key not set — cannot call Anthropic model")

        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = _system_with_cache(system, system_cache)

        gen = lf.start_generation(
            name=name, model=model_id, user_message=prompt,
            system=system, max_tokens=max_tokens,
        )

        last_error: BaseException | None = None
        for attempt in range(1, self._max_retries + 1):
            t0 = time.monotonic()
            try:
                msg = await self._anthropic.messages.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff_base ** (attempt - 1))
                    continue
                lf.end_generation_err(gen, exc, {"retries_exhausted": self._max_retries})
                raise

            duration = time.monotonic() - t0
            if not msg.content:
                # Treat empty content as retryable
                last_error = RuntimeError("empty content from Anthropic")
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff_base ** (attempt - 1))
                    continue
                lf.end_generation_err(gen, last_error)
                raise last_error

            text = msg.content[0].text
            usage = LLMUsage(
                input=msg.usage.input_tokens,
                output=msg.usage.output_tokens,
                cache_creation=getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
            )
            cost = lf.calc_cost(model_id, usage.input, usage.output)
            lf.end_generation_ok(
                gen,
                output_text=text,
                model_id=model_id,
                input_tokens=usage.input,
                output_tokens=usage.output,
                duration_sec=duration,
                attempt=attempt,
                cache_creation_input_tokens=usage.cache_creation,
                cache_read_input_tokens=usage.cache_read,
            )
            logger.info(
                "[llm.chat] OK model=%s in=%d out=%d cache_cr=%d cache_rd=%d "
                "cost=$%.6f dur=%.2fs attempt=%d",
                model_id, usage.input, usage.output,
                usage.cache_creation, usage.cache_read,
                cost, duration, attempt,
            )
            return LLMResponse(
                text=text, model=model_id, usage=usage, cost_usd=cost,
                duration_sec=duration, attempt=attempt, raw=msg,
            )

        # loop exhausted without returning — shouldn't happen but be explicit
        lf.end_generation_err(gen, last_error or RuntimeError("unknown"))
        raise last_error or RuntimeError("retries exhausted")

    async def _chat_deepseek(
        self,
        prompt: str,
        system: str | None,
        max_tokens: int,
        name: str,
    ) -> LLMResponse:
        if not self._deepseek_key:
            raise RuntimeError("deepseek_api_key not set — cannot call deepseek-chat")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        gen = lf.start_generation(
            name=name, model="deepseek-chat", user_message=prompt,
            system=system, max_tokens=max_tokens,
        )
        last_error: BaseException | None = None
        for attempt in range(1, self._max_retries + 1):
            t0 = time.monotonic()
            try:
                resp = await self._http.post(
                    _DEEPSEEK_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self._deepseek_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": self._temperature,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff_base ** (attempt - 1))
                    continue
                lf.end_generation_err(gen, exc, {"retries_exhausted": self._max_retries})
                raise

            duration = time.monotonic() - t0
            try:
                text = data["choices"][0]["message"]["content"]
                u = data.get("usage", {}) or {}
                usage = LLMUsage(
                    input=int(u.get("prompt_tokens", 0)),
                    output=int(u.get("completion_tokens", 0)),
                )
            except (KeyError, IndexError, TypeError) as exc:
                last_error = RuntimeError(f"malformed DeepSeek response: {exc}")
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff_base ** (attempt - 1))
                    continue
                lf.end_generation_err(gen, last_error)
                raise last_error from exc

            cost = lf.calc_cost("deepseek-chat", usage.input, usage.output)
            lf.end_generation_ok(
                gen, output_text=text, model_id="deepseek-chat",
                input_tokens=usage.input, output_tokens=usage.output,
                duration_sec=duration, attempt=attempt,
            )
            logger.info(
                "[llm.chat] DS OK in=%d out=%d cost=$%.6f dur=%.2fs attempt=%d",
                usage.input, usage.output, cost, duration, attempt,
            )
            return LLMResponse(
                text=text, model="deepseek-chat", usage=usage, cost_usd=cost,
                duration_sec=duration, attempt=attempt, raw=data,
            )

        lf.end_generation_err(gen, last_error or RuntimeError("unknown"))
        raise last_error or RuntimeError("retries exhausted")

    # ------------------------------------------------------------ agent turn
    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str = "sonnet",
        system: str | None = None,
        system_cache: bool = False,
        max_tokens: int = 4096,
        name: str = "llm.chat_with_tools",
    ) -> AgentTurn:
        """Run one ReAct turn: send messages + tools, return content + stop_reason.

        Anthropic-only. `messages` must follow the Messages API contract (a list
        of alternating user/assistant turns). `tools` is a list from
        `ToolRegistry.for_api()` or equivalent. The returned AgentTurn contains
        stop_reason, any tool_use blocks, text, and the raw content list that
        the caller must echo back as the next assistant message.
        """
        if self._anthropic is None:
            raise RuntimeError("anthropic_api_key not set — chat_with_tools requires Anthropic")

        model_id = MODEL_MAP.get(model, model)

        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": self._temperature,
            "messages": messages,
            "tools": tools,
        }
        if system is not None:
            kwargs["system"] = _system_with_cache(system, system_cache)

        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"),
            {"content": ""},
        )
        lf_user_msg = (
            last_user["content"] if isinstance(last_user.get("content"), str)
            else str(last_user.get("content", ""))[:1024]
        )
        gen = lf.start_generation(
            name=name, model=model_id, user_message=lf_user_msg,
            system=system, max_tokens=max_tokens,
            extra_params={"tools_count": len(tools)},
        )

        last_error: BaseException | None = None
        for attempt in range(1, self._max_retries + 1):
            t0 = time.monotonic()
            try:
                msg = await self._anthropic.messages.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff_base ** (attempt - 1))
                    continue
                lf.end_generation_err(gen, exc, {"retries_exhausted": self._max_retries})
                raise

            duration = time.monotonic() - t0

            tool_uses: list[ToolUse] = []
            text_parts: list[str] = []
            for block in msg.content:
                btype = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else None
                )
                if btype == "tool_use":
                    # SDK returns ToolUseBlock (pydantic); tests pass dicts.
                    # Empty input dict {} is falsy, so use hasattr/isinstance
                    # rather than `or`-fallback.
                    def _pick(obj: Any, key: str, default: Any) -> Any:
                        if hasattr(obj, key) and not isinstance(obj, dict):
                            return getattr(obj, key)
                        if isinstance(obj, dict):
                            return obj.get(key, default)
                        return default

                    tool_uses.append(ToolUse(
                        id=_pick(block, "id", ""),
                        name=_pick(block, "name", ""),
                        input=_pick(block, "input", {}),
                    ))
                elif btype == "text":
                    text_parts.append(getattr(block, "text", None) or (
                        block.get("text", "") if isinstance(block, dict) else ""
                    ))

            usage = LLMUsage(
                input=msg.usage.input_tokens,
                output=msg.usage.output_tokens,
                cache_creation=getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
            )
            cost = lf.calc_cost(model_id, usage.input, usage.output)

            lf.end_generation_ok(
                gen,
                output_text="".join(text_parts) or f"<{len(tool_uses)} tool_use(s)>",
                model_id=model_id,
                input_tokens=usage.input,
                output_tokens=usage.output,
                duration_sec=duration,
                attempt=attempt,
                cache_creation_input_tokens=usage.cache_creation,
                cache_read_input_tokens=usage.cache_read,
                extra_metadata={
                    "stop_reason": msg.stop_reason,
                    "tool_uses": len(tool_uses),
                },
            )
            logger.info(
                "[llm.tools] OK model=%s stop=%s tools=%d in=%d out=%d "
                "cost=$%.6f dur=%.2fs attempt=%d",
                model_id, msg.stop_reason, len(tool_uses),
                usage.input, usage.output, cost, duration, attempt,
            )
            return AgentTurn(
                model=model_id,
                stop_reason=msg.stop_reason,
                text="".join(text_parts),
                tool_uses=tool_uses,
                message_content=list(msg.content),
                usage=usage,
                cost_usd=cost,
                duration_sec=duration,
                attempt=attempt,
                raw=msg,
            )

        lf.end_generation_err(gen, last_error or RuntimeError("unknown"))
        raise last_error or RuntimeError("retries exhausted")

    # ------------------------------------------------------------ structured
    def _get_instructor(self) -> Any:
        if not _INSTRUCTOR_AVAILABLE:
            raise RuntimeError("instructor package not installed — cannot use chat_structured")
        if self._anthropic is None:
            raise RuntimeError("anthropic_api_key not set — chat_structured requires Anthropic")
        if self._instructor_client is None:
            self._instructor_client = instructor.from_anthropic(self._anthropic)
        return self._instructor_client

    async def chat_structured(
        self,
        prompt: str,
        response_model: type[T],
        model: str = "sonnet",
        system: str | None = None,
        system_cache: bool = False,
        max_tokens: int = 4096,
        name: str = "llm.chat_structured",
    ) -> tuple[T, LLMResponse]:
        """Structured output via instructor. Returns (pydantic_obj, raw LLMResponse).

        Anthropic only — DeepSeek via instructor would need openai client, not wired.
        """
        client = self._get_instructor()
        model_id = MODEL_MAP.get(model, model)

        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": prompt}],
            "response_model": response_model,
        }
        if system is not None:
            kwargs["system"] = _system_with_cache(system, system_cache)

        gen = lf.start_generation(
            name=name, model=model_id, user_message=prompt, system=system,
            max_tokens=max_tokens, extra_params={"structured": True},
        )

        last_error: BaseException | None = None
        for attempt in range(1, self._max_retries + 1):
            t0 = time.monotonic()
            try:
                result, raw = await client.messages.create_with_completion(**kwargs)
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff_base ** (attempt - 1))
                    continue
                lf.end_generation_err(gen, exc, {"retries_exhausted": self._max_retries})
                raise

            duration = time.monotonic() - t0
            u = raw.usage
            usage = LLMUsage(
                input=u.input_tokens,
                output=u.output_tokens,
                cache_creation=getattr(u, "cache_creation_input_tokens", 0) or 0,
                cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
            )
            cost = lf.calc_cost(model_id, usage.input, usage.output)

            lf.end_generation_ok(
                gen,
                output_text=result.model_dump(mode="json"),
                model_id=model_id,
                input_tokens=usage.input,
                output_tokens=usage.output,
                duration_sec=duration,
                attempt=attempt,
                cache_creation_input_tokens=usage.cache_creation,
                cache_read_input_tokens=usage.cache_read,
                extra_metadata={"response_model": response_model.__name__},
            )
            logger.info(
                "[llm.structured] OK model=%s schema=%s in=%d out=%d "
                "cost=$%.6f dur=%.2fs attempt=%d",
                model_id, response_model.__name__, usage.input, usage.output,
                cost, duration, attempt,
            )
            resp = LLMResponse(
                text="", model=model_id, usage=usage, cost_usd=cost,
                duration_sec=duration, attempt=attempt, raw=raw,
            )
            return result, resp

        lf.end_generation_err(gen, last_error or RuntimeError("unknown"))
        raise last_error or RuntimeError("retries exhausted")


def _system_with_cache(system: str, system_cache: bool) -> Any:
    """Wrap system prompt in cache_control block if requested, else raw string."""
    if not system_cache:
        return system
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
