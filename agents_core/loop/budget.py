"""Budget tripwires for the ReAct loop (research/architecture.md §3 and §9).

Hard limits that must trigger a clean stop before the agent burns unbounded
resources:

- max_steps          — turns per task (default 12)
- max_tokens_per_step — per-turn output tokens (default 2048)
- max_cost_usd       — total USD spent per task (default 0.50)
- max_duration_sec   — wall-clock per task (default 300s)

Usage:
    budget = BudgetTracker(BudgetLimits())
    budget.start()                    # record start time
    for step in loop:
        budget.record_step(tokens=turn.usage.output, cost=turn.cost_usd)
        budget.check()                # raises BudgetExceededError with reason

`check()` also raises when step count > max_steps even if nothing new has been
recorded — so you can call it unconditionally at the top of every loop
iteration. Every raise carries a `BudgetSnapshot` with totals + which limit
tripped, so the loop can emit a clean audit log entry before aborting.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

TripReason = Literal["max_steps", "max_tokens_per_step", "max_cost_usd", "max_duration_sec"]


@dataclass(frozen=True)
class BudgetLimits:
    max_steps: int = 12
    max_tokens_per_step: int = 2048
    max_cost_usd: float = 0.50
    max_duration_sec: float = 300.0


@dataclass(frozen=True)
class BudgetSnapshot:
    steps: int
    total_tokens: int
    total_cost_usd: float
    elapsed_sec: float
    last_step_tokens: int
    tripped_by: TripReason | None = None
    limits: BudgetLimits = field(default_factory=BudgetLimits)


class BudgetExceededError(RuntimeError):
    def __init__(self, snapshot: BudgetSnapshot) -> None:
        super().__init__(
            f"budget exceeded by {snapshot.tripped_by}: "
            f"steps={snapshot.steps}/{snapshot.limits.max_steps} "
            f"cost=${snapshot.total_cost_usd:.4f}/"
            f"${snapshot.limits.max_cost_usd:.2f} "
            f"elapsed={snapshot.elapsed_sec:.1f}s/"
            f"{snapshot.limits.max_duration_sec:.0f}s "
            f"last_step_tokens={snapshot.last_step_tokens}/"
            f"{snapshot.limits.max_tokens_per_step}"
        )
        self.snapshot = snapshot


@dataclass
class _AuditEntry:
    step: int
    tokens: int
    cost: float
    elapsed: float


class BudgetTracker:
    """Mutable per-task tracker. Create a new one for each task."""

    def __init__(self, limits: BudgetLimits | None = None) -> None:
        self._limits = limits or BudgetLimits()
        self._steps = 0
        self._total_tokens = 0
        self._total_cost_usd = 0.0
        self._last_step_tokens = 0
        self._start_mono: float | None = None
        self._audit: list[_AuditEntry] = []

    @property
    def limits(self) -> BudgetLimits:
        return self._limits

    @property
    def audit(self) -> list[_AuditEntry]:
        return list(self._audit)

    def start(self) -> None:
        self._start_mono = time.monotonic()

    def _elapsed(self) -> float:
        if self._start_mono is None:
            return 0.0
        return time.monotonic() - self._start_mono

    def record_step(self, *, tokens: int, cost: float) -> None:
        if self._start_mono is None:
            self.start()
        self._steps += 1
        self._total_tokens += tokens
        self._total_cost_usd += cost
        self._last_step_tokens = tokens
        self._audit.append(_AuditEntry(
            step=self._steps,
            tokens=tokens,
            cost=cost,
            elapsed=self._elapsed(),
        ))

    def snapshot(self, tripped_by: TripReason | None = None) -> BudgetSnapshot:
        return BudgetSnapshot(
            steps=self._steps,
            total_tokens=self._total_tokens,
            total_cost_usd=self._total_cost_usd,
            elapsed_sec=self._elapsed(),
            last_step_tokens=self._last_step_tokens,
            tripped_by=tripped_by,
            limits=self._limits,
        )

    def check(self) -> None:
        """Raise BudgetExceededError if any tripwire fired. Idempotent — safe at top of loop."""
        reason: TripReason | None = None

        if self._steps >= self._limits.max_steps:
            reason = "max_steps"
        elif self._last_step_tokens > self._limits.max_tokens_per_step:
            reason = "max_tokens_per_step"
        elif self._total_cost_usd > self._limits.max_cost_usd:
            reason = "max_cost_usd"
        elif self._elapsed() > self._limits.max_duration_sec:
            reason = "max_duration_sec"

        if reason is not None:
            raise BudgetExceededError(self.snapshot(tripped_by=reason))
