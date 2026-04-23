"""Aggregate metrics over a batch of trajectories (research/evaluation.md §1).

Inputs are tuples (LoopResult, JudgeResult) — the judge decides success,
these functions just tabulate. Pure functions, no IO, no LLM.

Metrics (research §1 Level 1 + Level 2):
- ``success_rate`` — share of trials where Judge returned success.
- ``cost_per_success`` — total $ / successful trials (``inf`` if zero).
- ``steps_to_success`` — average step count of successful trials.
- ``tool_error_rate`` — total errors / total tool calls across all trials.
- ``learning_curve`` — success_rate per trial_n (reflexion).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from agents_core.evaluation.judge import Judge, JudgeResult
from agents_core.loop.react import LoopResult

__all__ = ["MetricsSummary", "summarize", "learning_curve"]

Trial = tuple[LoopResult, JudgeResult]


@dataclass(frozen=True)
class MetricsSummary:
    n_trials: int
    n_success: int
    success_rate: float  # 0..1
    cost_per_success_usd: float  # inf if n_success == 0
    avg_steps_to_success: float  # NaN if n_success == 0
    tool_error_rate: float  # 0..1


def summarize(trials: list[Trial]) -> MetricsSummary:
    if not trials:
        return MetricsSummary(0, 0, 0.0, math.inf, math.nan, 0.0)

    n_trials = len(trials)
    n_success = sum(1 for _, j in trials if j.success)
    total_cost = sum(r.cost_usd for r, _ in trials)

    success_steps = [r.step_count for r, j in trials if j.success]
    avg_steps = (sum(success_steps) / len(success_steps)) if success_steps else math.nan

    errs = 0
    total_calls = 0
    for r, _ in trials:
        e, t = Judge.tool_error_rate(r)
        errs += e
        total_calls += t
    tool_err = (errs / total_calls) if total_calls else 0.0

    return MetricsSummary(
        n_trials=n_trials,
        n_success=n_success,
        success_rate=n_success / n_trials,
        cost_per_success_usd=(total_cost / n_success) if n_success else math.inf,
        avg_steps_to_success=avg_steps,
        tool_error_rate=tool_err,
    )


def learning_curve(trials_by_trial_n: dict[int, list[Trial]]) -> dict[int, float]:
    """Return ``{trial_n: success_rate}``.

    Use this to track whether Reflexion is actually helping: success_rate
    should grow from trial 1 to later retries. Research §1: expect
    trial_1 → trial_5 gain ≥ 15 p.p., otherwise the lessons aren't working.
    """
    curve: dict[int, float] = {}
    for n, bucket in trials_by_trial_n.items():
        if not bucket:
            continue
        success = sum(1 for _, j in bucket if j.success)
        curve[n] = success / len(bucket)
    return curve


def bucket_by_trial(
    trials: list[Trial],
    trial_n_of: list[int] | None = None,
) -> dict[int, list[Trial]]:
    """Helper: bucket a flat trial list by trial_n.

    ``trial_n_of[i]`` is the 1-based retry index for trial i. If omitted,
    every trial is bucket 1 (no retries). Kept separate from the Trial
    tuple so we don't force callers to re-wire existing LoopResult data.
    """
    if trial_n_of is None:
        return {1: list(trials)}
    if len(trial_n_of) != len(trials):
        raise ValueError("trial_n_of must align with trials (same length)")
    out: dict[int, list[Trial]] = defaultdict(list)
    for t, n in zip(trials, trial_n_of, strict=True):
        out[n].append(t)
    return dict(out)
