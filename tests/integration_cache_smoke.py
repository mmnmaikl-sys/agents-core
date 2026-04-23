"""Task 0.4 DoD smoke — verify prompt caching helper produces cache_read > 0 on 2nd call.

Uses a long (>1024 tokens) stable system prompt, calls LLMClient.chat twice
with system_cache=True. First call: cache_creation > 0 (cache written).
Second call: cache_read > 0 (cache hit).

Requires: ANTHROPIC_API_KEY, LANGFUSE_* (optional).
"""
from __future__ import annotations

import asyncio
import os
import sys

from agents_core.llm import LLMClient

# ~2500 tokens of stable filler content. Must be identical across both calls.
SYSTEM_PROMPT = (
    "Ты — эталонный ассистент для тестирования prompt caching.\n\n"
    + "### Контекст системы\n" * 50
    + "\n\n".join(
        f"Пункт {i}: это искусственно длинный текст чтобы превысить минимальный порог "
        f"Anthropic prompt caching (1024 токена). В реальности сюда пойдут юридические "
        f"правила, инструкции агента, каталог инструментов и знания из RAG. Пункт {i} "
        f"нужен только чтобы система prompt достигла объёма при котором Anthropic готов "
        f"записать его в ephemeral cache. Пункт {i} повторяет себя для длины."
        for i in range(60)
    )
    + "\n\nКонец системного промпта. Отвечай строго одним словом: 'ok'."
)


async def main() -> int:
    ak = os.environ.get("ANTHROPIC_API_KEY")
    if not ak:
        print("[cache-smoke] ANTHROPIC_API_KEY missing", file=sys.stderr)
        return 2

    print(f"[cache-smoke] system prompt len={len(SYSTEM_PROMPT)} chars (~{len(SYSTEM_PROMPT)//4} tokens)")

    async with LLMClient(anthropic_api_key=ak) as client:
        r1 = await client.chat(
            prompt="Скажи ok.",
            model="haiku",
            system=SYSTEM_PROMPT,
            system_cache=True,
            max_tokens=8,
            name="agents_core.cache_smoke.1",
        )
        print(
            f"[cache-smoke] call 1: in={r1.usage.input} out={r1.usage.output} "
            f"cache_cr={r1.usage.cache_creation} cache_rd={r1.usage.cache_read}"
        )

        r2 = await client.chat(
            prompt="Скажи ok ещё раз.",
            model="haiku",
            system=SYSTEM_PROMPT,
            system_cache=True,
            max_tokens=8,
            name="agents_core.cache_smoke.2",
        )
        print(
            f"[cache-smoke] call 2: in={r2.usage.input} out={r2.usage.output} "
            f"cache_cr={r2.usage.cache_creation} cache_rd={r2.usage.cache_read}"
        )

        assert r1.usage.cache_creation > 0, (
            f"call 1 should have cache_creation > 0, got {r1.usage.cache_creation}"
        )
        assert r2.usage.cache_read > 0, (
            f"call 2 should have cache_read > 0, got {r2.usage.cache_read}"
        )

    try:
        from agents_core.observability import langfuse_wrap

        lf = langfuse_wrap.get_langfuse()
        if lf is not None:
            lf.flush()
            print("[cache-smoke] Langfuse flushed — check traces agents_core.cache_smoke.{1,2}")
    except Exception as e:  # noqa: BLE001
        print(f"[cache-smoke] Langfuse flush skipped: {e}")

    print("[cache-smoke] OK — cache_read > 0 on 2nd call, DoD 0.4 satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
