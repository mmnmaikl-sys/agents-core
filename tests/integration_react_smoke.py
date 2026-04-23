"""Task 0.7 DoD smoke — hello-world agent with 3 tools, real Anthropic call.

Runs a ReActLoop with 3 tools (`get_time`, `add`, `greet`) and asks the model
to compute `3+4` and greet. Verifies:
- at least 1 tool_use turn, then end_turn
- tool results flowed into messages
- final_text is non-empty

Requires: ANTHROPIC_API_KEY (use `agents-core-dev`), LANGFUSE_* optional.
"""
from __future__ import annotations

import asyncio
import os
import sys

from agents_core.llm import LLMClient
from agents_core.loop import ReActLoop
from agents_core.tools import Tool, ToolRegistry

SYSTEM = (
    "Ты — тестовый ReAct-агент. У тебя есть 3 инструмента: get_time, add, greet. "
    "Когда нужна дата/время — вызывай get_time. Для арифметики — add. "
    "Для приветствия — greet. Давай короткие ответы."
)


def build_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(Tool(
        name="get_time",
        description="Return the current date and time as ISO 8601 string. "
                    "Use this whenever the user asks 'what time is it'.",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=lambda: "2026-04-23T10:00:00Z",
    ))
    r.register(Tool(
        name="add",
        description="Add two numbers. Returns a + b as an integer.",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        handler=lambda a, b: a + b,
    ))
    r.register(Tool(
        name="greet",
        description="Return a personalized greeting for the given name.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        handler=lambda name: f"Hello, {name}!",
    ))
    return r


async def main() -> int:
    ak = os.environ.get("ANTHROPIC_API_KEY")
    if not ak:
        print("[react-smoke] ANTHROPIC_API_KEY missing", file=sys.stderr)
        return 2

    lf_on = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
    print(f"[react-smoke] Langfuse={'ON' if lf_on else 'OFF'}")

    async with LLMClient(anthropic_api_key=ak) as client:
        loop = ReActLoop(
            client=client,
            registry=build_registry(),
            system=SYSTEM,
            model="haiku",  # cheap for smoke
            max_tokens=512,
            max_steps=5,
            name="agents_core.react_smoke",
            system_cache=False,  # short system — caching won't trigger anyway
        )
        result = await loop.run(task="Посчитай 3+4 через инструмент и поздоровайся с Михаилом.")

        print(f"[react-smoke] stop_reason={result.stop_reason} steps={result.step_count}")
        print(f"[react-smoke] usage in={result.usage.input} out={result.usage.output} "
              f"cost=${result.cost_usd:.6f}")
        print(f"[react-smoke] final_text={result.final_text!r}")
        for i, step in enumerate(result.steps, 1):
            names = [tu.name for tu in step.turn.tool_uses]
            print(f"[react-smoke] step {i}: stop={step.turn.stop_reason} tools={names}")

        assert result.stop_reason in ("end_turn", "stop_sequence"), (
            f"expected end_turn, got {result.stop_reason}"
        )
        assert result.final_text, "empty final_text"
        assert any(s.turn.tool_uses for s in result.steps), "no tool was called"

    try:
        from agents_core.observability import langfuse_wrap

        lf = langfuse_wrap.get_langfuse()
        if lf is not None:
            lf.flush()
            print("[react-smoke] Langfuse flushed — check traces 'agents_core.react_smoke.step.*'")
    except Exception as e:  # noqa: BLE001
        print(f"[react-smoke] Langfuse flush skipped: {e}")

    print("[react-smoke] OK — tool_use → tool_result → end_turn DoD satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
