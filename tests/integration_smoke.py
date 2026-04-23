"""Integration smoke for agents_core.llm.client — NOT run by default pytest.

Requires env vars:
    ANTHROPIC_API_KEY
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    (optional) LANGFUSE_HOST

Usage:
    LANGFUSE_HOST=https://cloud.langfuse.com \\
    LANGFUSE_PUBLIC_KEY=... \\
    LANGFUSE_SECRET_KEY=... \\
    ANTHROPIC_API_KEY=... \\
    python tests/integration_smoke.py

Expected output:
    1. Real Haiku call succeeds, resp.text printed
    2. Cost > 0, input_tokens > 0
    3. Langfuse trace URL printed (goto to verify in UI)
"""
from __future__ import annotations

import asyncio
import os
import sys

from agents_core.llm import LLMClient


async def main() -> int:
    ak = os.environ.get("ANTHROPIC_API_KEY")
    if not ak:
        print("[smoke] ANTHROPIC_API_KEY missing — skip", file=sys.stderr)
        return 2

    lf_pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    lf_sk = os.environ.get("LANGFUSE_SECRET_KEY")
    print(f"[smoke] Langfuse={'ON' if (lf_pk and lf_sk) else 'OFF'}")

    async with LLMClient(anthropic_api_key=ak) as client:
        resp = await client.chat(
            prompt="Say 'smoke ok' in exactly those two words.",
            model="haiku",
            max_tokens=32,
            name="agents_core.smoke",
        )
        print(f"[smoke] model={resp.model}")
        print(f"[smoke] text={resp.text!r}")
        print(f"[smoke] usage=in{resp.usage.input}/out{resp.usage.output} "
              f"cache_cr{resp.usage.cache_creation}/rd{resp.usage.cache_read}")
        print(
            f"[smoke] cost=${resp.cost_usd:.6f} "
            f"dur={resp.duration_sec:.2f}s attempt={resp.attempt}"
        )

        assert resp.text, "empty text"
        assert resp.usage.input > 0, "input_tokens == 0"
        assert resp.cost_usd > 0, "cost == 0"

    # Force Langfuse flush so the trace is visible immediately
    try:
        from agents_core.observability import langfuse_wrap

        lf = langfuse_wrap.get_langfuse()
        if lf is not None:
            lf.flush()
            print("[smoke] Langfuse flushed — check cloud.langfuse.com for 'agents_core.smoke'")
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] Langfuse flush skipped: {e}")

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
