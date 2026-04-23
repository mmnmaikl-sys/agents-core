"""Unit tests for agents_core.loop.budget.

DoD: simulate each tripwire being exceeded → raise BudgetExceededError +
audit log captured.

Covers:
- max_steps — raises once steps reach limit (next check on/after)
- max_tokens_per_step — single step > per-step cap
- max_cost_usd — total cost crossing threshold
- max_duration_sec — elapsed wall-clock exceeding deadline
- happy path — all below limits, check() is noop
- BudgetSnapshot carries limits, tripped_by, totals
- audit trail preserved
- start() idempotent via record_step
"""
from __future__ import annotations

import time

import pytest

from agents_core.loop.budget import (
    BudgetExceededError,
    BudgetLimits,
    BudgetTracker,
)


# ---------------------------------------------------------- happy path
def test_check_noop_when_under_limits():
    t = BudgetTracker(BudgetLimits(max_steps=5, max_cost_usd=1.0))
    t.start()
    t.record_step(tokens=100, cost=0.01)
    t.check()  # should not raise
    snap = t.snapshot()
    assert snap.steps == 1
    assert snap.total_cost_usd == pytest.approx(0.01)
    assert snap.tripped_by is None


def test_snapshot_default_values():
    t = BudgetTracker()
    snap = t.snapshot()
    assert snap.steps == 0
    assert snap.total_cost_usd == 0.0
    assert snap.elapsed_sec == 0.0  # start() not called
    assert snap.tripped_by is None


# ---------------------------------------------------------- max_steps
def test_tripwire_max_steps():
    t = BudgetTracker(BudgetLimits(max_steps=2, max_cost_usd=100, max_tokens_per_step=10_000))
    t.start()
    t.record_step(tokens=10, cost=0.001)
    t.check()
    t.record_step(tokens=10, cost=0.001)
    with pytest.raises(BudgetExceededError) as exc:
        t.check()
    assert exc.value.snapshot.tripped_by == "max_steps"
    assert exc.value.snapshot.steps == 2
    assert "max_steps" in str(exc.value)


# ---------------------------------------------------------- max_tokens_per_step
def test_tripwire_max_tokens_per_step():
    t = BudgetTracker(BudgetLimits(max_tokens_per_step=100))
    t.start()
    t.record_step(tokens=150, cost=0.001)
    with pytest.raises(BudgetExceededError) as exc:
        t.check()
    assert exc.value.snapshot.tripped_by == "max_tokens_per_step"
    assert exc.value.snapshot.last_step_tokens == 150


# ---------------------------------------------------------- max_cost_usd
def test_tripwire_max_cost_usd():
    t = BudgetTracker(BudgetLimits(max_cost_usd=0.05))
    t.start()
    t.record_step(tokens=10, cost=0.02)
    t.record_step(tokens=10, cost=0.02)
    t.check()  # 0.04 — still ok
    t.record_step(tokens=10, cost=0.02)  # total 0.06 > 0.05
    with pytest.raises(BudgetExceededError) as exc:
        t.check()
    assert exc.value.snapshot.tripped_by == "max_cost_usd"
    assert exc.value.snapshot.total_cost_usd == pytest.approx(0.06)


# ---------------------------------------------------------- max_duration_sec
def test_tripwire_max_duration_sec(monkeypatch):
    t = BudgetTracker(BudgetLimits(max_duration_sec=0.05, max_steps=100))

    mono_now = [1000.0]

    def fake_mono():
        return mono_now[0]

    monkeypatch.setattr(time, "monotonic", fake_mono)

    t.start()
    t.record_step(tokens=10, cost=0.001)
    mono_now[0] += 0.1  # advance 100ms
    with pytest.raises(BudgetExceededError) as exc:
        t.check()
    assert exc.value.snapshot.tripped_by == "max_duration_sec"
    assert exc.value.snapshot.elapsed_sec >= 0.05


# ---------------------------------------------------------- audit log
def test_audit_log_preserved_across_steps():
    t = BudgetTracker()
    t.start()
    t.record_step(tokens=10, cost=0.001)
    t.record_step(tokens=20, cost=0.002)
    t.record_step(tokens=30, cost=0.003)

    audit = t.audit
    assert len(audit) == 3
    assert audit[0].step == 1 and audit[0].tokens == 10
    assert audit[1].step == 2 and audit[1].tokens == 20
    assert audit[2].step == 3 and audit[2].tokens == 30
    # audit is a copy — external mutation doesn't affect internal state
    audit.clear()
    assert len(t.audit) == 3


# ---------------------------------------------------------- start() semantics
def test_record_step_auto_starts():
    t = BudgetTracker()
    # never called start() — record_step should still work
    t.record_step(tokens=10, cost=0.001)
    t.check()
    assert t.snapshot().elapsed_sec >= 0


def test_snapshot_includes_limits():
    limits = BudgetLimits(max_steps=7, max_cost_usd=1.23)
    t = BudgetTracker(limits)
    snap = t.snapshot(tripped_by="max_steps")
    assert snap.limits is limits
    assert snap.tripped_by == "max_steps"


# ---------------------------------------------------------- integration with ReActLoop (smoke)
@pytest.mark.asyncio
async def test_tracker_usable_around_react_loop():
    """Demonstrate the intended wiring: tracker around a mocked loop."""
    from unittest.mock import AsyncMock, MagicMock

    from agents_core.llm.client import AgentTurn, LLMUsage, ToolUse
    from agents_core.loop import ReActLoop
    from agents_core.tools import Tool, ToolRegistry

    registry = ToolRegistry()
    registry.register(Tool(
        name="noop",
        description="Do nothing",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=lambda: "ok",
    ))

    def _turn(stop: str, text="", tus=None) -> AgentTurn:
        return AgentTurn(
            model="m", stop_reason=stop, text=text, tool_uses=tus or [],
            message_content=[], usage=LLMUsage(input=10, output=5),
            cost_usd=0.001,
        )

    client = MagicMock()
    client.chat_with_tools = AsyncMock(side_effect=[
        _turn("tool_use", tus=[ToolUse(id="a", name="noop", input={})]),
        _turn("end_turn", text="done"),
    ])

    budget = BudgetTracker(BudgetLimits(max_steps=5, max_cost_usd=1.0))
    budget.start()

    loop = ReActLoop(client, registry, system="s")
    result = await loop.run(task="t")

    # Simulate the wiring: user code would call these inside loop — here we
    # replay after the fact to verify the tracker accumulates correctly.
    for step in result.steps:
        budget.record_step(tokens=step.turn.usage.output, cost=step.turn.cost_usd)
        budget.check()

    assert budget.snapshot().steps == 2
    assert budget.snapshot().total_cost_usd == pytest.approx(0.002)
