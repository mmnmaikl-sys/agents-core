"""Prompt caching helpers for Anthropic.

Two use cases covered:

1. **split_input_block** — templates that already embed dynamic data inside
   an `<input>...</input>` block. Extracts it so the rest of the prompt
   (cache-stable) can be sent as system= with cache_control, and the dynamic
   block goes into the user message.

2. **wrap_system_with_cache_control** — wraps a raw system prompt into the
   Anthropic content-block list with cache_control=ephemeral. Used when the
   caller already knows the prompt is stable enough to cache (system prompts,
   long knowledge dumps, tool catalogs).

Minimum cache block size: 1024 tokens (Sonnet/Haiku 4.5). The helpers do not
enforce this — the caller is responsible for ensuring the text is long enough
to be cached. If it's too short Anthropic silently sends cache_creation=0,
cache_read=0 and you're back to normal pricing with no hit.

Ported from speech-analytics/src/steps/_prompt_split.py (2026-04-23).
"""
from __future__ import annotations

import re
from typing import Any

_INPUT_TAG_RE = re.compile(r"<input>.*?</input>", re.DOTALL)


def split_input_block(rendered_prompt: str) -> tuple[str, str]:
    """Return (static_instructions, dynamic_input_block).

    static_instructions: rendered_prompt with the `<input>...</input>` block
        removed (whitespace trimmed). Stable across calls that share the
        same template — send it as system= with cache_control.
    dynamic_input_block: the full `<input>...</input>` block (with tags),
        or the original prompt if no `<input>` tag is present.

    If no <input> block is found, returns ("", rendered_prompt) so the
    caller can fall back to the non-cached path safely.
    """
    match = _INPUT_TAG_RE.search(rendered_prompt)
    if match is None:
        return "", rendered_prompt
    static = _INPUT_TAG_RE.sub("", rendered_prompt).strip()
    dynamic = match.group(0)
    return static, dynamic


def wrap_system_with_cache_control(text: str) -> list[dict[str, Any]]:
    """Wrap a system prompt into Anthropic's content-block form with cache_control.

    Returns a list with a single text block marked cache_control=ephemeral.
    Use as:
        client.messages.create(
            system=wrap_system_with_cache_control(SYSTEM_PROMPT),
            ...
        )

    LLMClient.chat also accepts `system_cache=True` + plain str — prefer that.
    This helper is for callers that build the Anthropic request manually.
    """
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
