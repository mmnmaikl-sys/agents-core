"""Canonical Anthropic ReAct loop (research/architecture.md §3 layer 3).

The loop:
  messages := [user: task]
  while step < max_steps:
      turn := client.chat_with_tools(messages, tools, system, ...)
      if turn.stop_reason == "end_turn":
          return turn.text
      if turn.stop_reason == "tool_use":
          tool_results := parallel(registry.call(tu) for tu in turn.tool_uses)
          messages += [assistant: turn.message_content, user: tool_results]

- Parallel tool use: all tool_use blocks from one turn run through
  asyncio.gather (research §3 and Anthropic cookbook "parallel tool use").
- Tool errors become is_error=true tool_result blocks (research §9 rule 5),
  the model decides whether to retry, branch, or escalate.
- Budget tripwires (max_steps, max_cost, max_tokens/step, timeout) plus
  anti-infinite-loop detection live in `loop.budget` and `loop.dedup` —
  this module takes pluggable hooks to keep concerns separated.

`hello_agent.py` (task 0.25) instantiates a `ReActLoop` with 3 tools and
runs a single tool_use→tool_result→end_turn cycle; that is the DoD here.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agents_core.llm.client import AgentTurn, LLMClient, LLMUsage
from agents_core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_DEFAULT_MAX_STEPS = 12


@dataclass
class StepRecord:
    """One loop step — the model's turn + tool results we fed back (if any)."""

    step: int
    turn: AgentTurn
    tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LoopResult:
    final_text: str
    stop_reason: str
    steps: list[StepRecord]
    messages: list[dict[str, Any]]
    usage: LLMUsage
    cost_usd: float

    @property
    def step_count(self) -> int:
        return len(self.steps)


class MaxStepsExceededError(RuntimeError):
    """Raised when the loop hits max_steps without an end_turn."""


class ReActLoop:
    """ReAct loop wrapper. Stateless — create a new instance per task.

    Parameters
    ----------
    client:
        Configured LLMClient.
    registry:
        ToolRegistry that exposes the tools to the model. `registry.for_api()`
        is called once per turn — if you need tier/tag filtering, pre-build
        a filtered sub-registry or subclass and override `_tools_for_turn`.
    system:
        System prompt (canonical prompt is in templates/canonical.txt for
        real agents; tests pass a short one).
    model / max_tokens / system_cache:
        Forwarded to `client.chat_with_tools`.
    max_steps:
        Hard stop on number of model turns (default 12, research §3 and §9).
    name:
        Langfuse span prefix (each turn uses `{name}.step.{n}`).
    """

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        system: str,
        model: str = "sonnet",
        max_tokens: int = 4096,
        system_cache: bool = True,
        max_steps: int = _DEFAULT_MAX_STEPS,
        name: str = "react",
    ) -> None:
        self._client = client
        self._registry = registry
        self._system = system
        self._model = model
        self._max_tokens = max_tokens
        self._system_cache = system_cache
        self._max_steps = max_steps
        self._name = name

    def _tools_for_turn(self) -> list[dict[str, Any]]:
        return self._registry.for_api()

    async def run(
        self,
        task: str,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> LoopResult:
        messages: list[dict[str, Any]] = list(extra_messages or [])
        messages.append({"role": "user", "content": task})

        steps: list[StepRecord] = []
        agg = LLMUsage()
        agg_cost = 0.0

        for step_i in range(1, self._max_steps + 1):
            turn = await self._client.chat_with_tools(
                messages=messages,
                tools=self._tools_for_turn(),
                model=self._model,
                system=self._system,
                system_cache=self._system_cache,
                max_tokens=self._max_tokens,
                name=f"{self._name}.step.{step_i}",
            )

            agg.input += turn.usage.input
            agg.output += turn.usage.output
            agg.cache_creation += turn.usage.cache_creation
            agg.cache_read += turn.usage.cache_read
            agg_cost += turn.cost_usd

            record = StepRecord(step=step_i, turn=turn)
            steps.append(record)

            if turn.stop_reason == "end_turn" or turn.stop_reason == "stop_sequence":
                logger.info(
                    "[react] %s finished at step %d stop=%s cost=$%.6f",
                    self._name, step_i, turn.stop_reason, agg_cost,
                )
                return LoopResult(
                    final_text=turn.text,
                    stop_reason=turn.stop_reason,
                    steps=steps,
                    messages=messages,
                    usage=agg,
                    cost_usd=agg_cost,
                )

            if turn.stop_reason == "tool_use":
                tool_results = await self._execute_tools(turn)
                record.tool_results = tool_results
                messages.append({"role": "assistant", "content": turn.message_content})
                messages.append({"role": "user", "content": tool_results})
                continue

            # max_tokens or unknown — bail out with what we have
            logger.warning(
                "[react] %s step %d stop=%s — exiting with partial result",
                self._name, step_i, turn.stop_reason,
            )
            return LoopResult(
                final_text=turn.text,
                stop_reason=turn.stop_reason,
                steps=steps,
                messages=messages,
                usage=agg,
                cost_usd=agg_cost,
            )

        raise MaxStepsExceededError(
            f"{self._name} exceeded max_steps={self._max_steps} without end_turn",
        )

    async def _execute_tools(self, turn: AgentTurn) -> list[dict[str, Any]]:
        """Run all tool_use blocks in parallel, wrap errors as is_error results."""
        coros = [self._execute_one(tu.id, tu.name, tu.input) for tu in turn.tool_uses]
        return await asyncio.gather(*coros)

    async def _execute_one(
        self, tool_use_id: str, tool_name: str, tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            result = await self._registry.call(tool_name, **tool_input)
        except Exception as exc:  # noqa: BLE001 — want everything, model decides
            logger.warning("[react] tool %s raised: %s: %s", tool_name, type(exc).__name__, exc)
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"{type(exc).__name__}: {exc}",
                "is_error": True,
            }
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": _serialize_tool_result(result),
            "is_error": False,
        }


def _serialize_tool_result(value: Any) -> str:
    """Render any Python value as a string the model can read."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
