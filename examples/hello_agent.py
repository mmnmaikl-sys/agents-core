"""Hello-world agents-core demo (task 0.25).

Reference agent: 5 tools, one ReAct run, real Anthropic call. Run:

    ANTHROPIC_API_KEY=... python examples/hello_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime

from agents_core.llm import LLMClient
from agents_core.loop import ReActLoop
from agents_core.templates import CANONICAL_SYSTEM_PROMPT
from agents_core.tools import Tool, ToolRegistry

_OBJ: dict = {"type": "object", "properties": {}, "required": []}
_NUM = {"type": "object", "properties": {"a": {"type": "number"},
                                         "b": {"type": "number"}}, "required": ["a", "b"]}
_STR = {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]}
_NAME = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
_TEXT = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def _tools() -> ToolRegistry:
    return ToolRegistry([
        Tool("now", "Current ISO-8601 UTC time.", _OBJ,
             lambda: datetime.utcnow().isoformat()),
        Tool("add", "Add two numbers.", _NUM, lambda a, b: a + b),
        Tool("greet", "Greet someone by name.", _NAME,
             lambda name: f"Здравствуйте, {name}!"),
        Tool("word_count", "Count words in a text.", _TEXT,
             lambda text: len(text.split())),
        Tool("reverse", "Reverse a string.", _STR, lambda s: s[::-1]),
    ])


async def main() -> int:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC_API_KEY required", file=sys.stderr)
        return 2
    async with LLMClient(anthropic_api_key=key) as client:
        loop = ReActLoop(
            client, _tools(),
            system=CANONICAL_SYSTEM_PROMPT,
            model="haiku", max_steps=4,
        )
        result = await loop.run(
            "Посчитай 12+30, скажи текущее время и поздоровайся с Михаилом."
        )
    print(f"answer: {result.final_text}")
    print(f"steps={result.step_count} cost=${result.cost_usd:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
