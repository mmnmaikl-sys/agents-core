"""Unit tests for agents_core.evaluation.metrics (task 0.13)."""

from __future__ import annotations

import math

import pytest

from agents_core.evaluation import (
    JudgeResult,
    bucket_by_trial,
    learning_curve,
    summarize,
)
from agents_core.llm.client import AgentTurn, LLMUsage
from agents_core.loop.react import LoopResult, StepRecord


def _trial(
    success: bool,
    *,
    steps: int = 2,
    cost: float = 0.01,
    tool_calls: int = 0,
    tool_errors: int = 0,
) -> tuple[LoopResult, JudgeResult]:
    tr_per_step: list[dict] = []
    for i in range(tool_calls):
        tr_per_step.append({"type": "tool_result", "is_error": i < tool_errors})
    turn = AgentTurn(
        model="claude-haiku-4-5-20251001",
        stop_reason="end_turn",
        text="",
        tool_uses=[],
        message_content=[],
        usage=LLMUsage(),
        cost_usd=0.0,
        duration_sec=0.0,
    )
    step_records = []
    for i in range(steps):
        # Put all tool_results on the last step so the total matches.
        tr = tr_per_step if i == steps - 1 else []
        step_records.append(StepRecord(step=i + 1, turn=turn, tool_results=list(tr)))
    loop = LoopResult(
        final_text="",
        stop_reason="end_turn",
        steps=step_records,
        messages=[],
        usage=LLMUsage(),
        cost_usd=cost,
    )
    judge = JudgeResult(success=success, signals={"finish": True}, reason="")
    return loop, judge


class TestSummarize:
    def test_empty_trials_is_zero_but_safe(self):
        s = summarize([])
        assert s.n_trials == 0
        assert s.n_success == 0
        assert s.success_rate == 0.0
        assert s.cost_per_success_usd == math.inf
        assert math.isnan(s.avg_steps_to_success)
        assert s.tool_error_rate == 0.0

    def test_all_successes(self):
        trials = [
            _trial(True, steps=2, cost=0.01),
            _trial(True, steps=4, cost=0.02),
        ]
        s = summarize(trials)
        assert s.n_trials == 2
        assert s.n_success == 2
        assert s.success_rate == 1.0
        assert s.cost_per_success_usd == pytest.approx(0.015)
        assert s.avg_steps_to_success == pytest.approx(3.0)
        assert s.tool_error_rate == 0.0

    def test_mixed_with_errors(self):
        trials = [
            _trial(True, steps=2, cost=0.01, tool_calls=2, tool_errors=0),
            _trial(False, steps=8, cost=0.05, tool_calls=3, tool_errors=2),
            _trial(True, steps=3, cost=0.02, tool_calls=1, tool_errors=1),
        ]
        s = summarize(trials)
        assert s.n_trials == 3
        assert s.n_success == 2
        assert s.success_rate == pytest.approx(2 / 3)
        # cost_per_success uses total cost / n_success
        assert s.cost_per_success_usd == pytest.approx((0.01 + 0.05 + 0.02) / 2)
        # avg_steps uses only successful trials: (2 + 3) / 2
        assert s.avg_steps_to_success == pytest.approx(2.5)
        # tool_error_rate aggregates across ALL trials (success or not)
        assert s.tool_error_rate == pytest.approx(3 / 6)

    def test_all_failures_inf_cost(self):
        trials = [_trial(False, cost=0.1), _trial(False, cost=0.2)]
        s = summarize(trials)
        assert s.cost_per_success_usd == math.inf
        assert math.isnan(s.avg_steps_to_success)
        assert s.success_rate == 0.0


class TestLearningCurve:
    def test_empty_returns_empty(self):
        assert learning_curve({}) == {}

    def test_monotonic_improvement(self):
        # trial 1: 30% success; trial 2: 60%; trial 3: 90%
        buckets = {
            1: [_trial(i < 3) for i in range(10)],
            2: [_trial(i < 6) for i in range(10)],
            3: [_trial(i < 9) for i in range(10)],
        }
        curve = learning_curve(buckets)
        assert curve == {1: 0.3, 2: 0.6, 3: 0.9}

    def test_skips_empty_buckets(self):
        buckets = {1: [_trial(True)], 2: []}
        curve = learning_curve(buckets)
        assert curve == {1: 1.0}


class TestBucketByTrial:
    def test_default_bucket_one(self):
        trials = [_trial(True), _trial(False)]
        assert bucket_by_trial(trials) == {1: trials}

    def test_custom_trial_n_map(self):
        a, b, c = _trial(True), _trial(False), _trial(True)
        out = bucket_by_trial([a, b, c], trial_n_of=[1, 1, 2])
        assert out == {1: [a, b], 2: [c]}

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            bucket_by_trial([_trial(True), _trial(False)], trial_n_of=[1])

    def test_integration_with_learning_curve(self):
        a, b, c, d = _trial(True), _trial(False), _trial(True), _trial(True)
        buckets = bucket_by_trial([a, b, c, d], trial_n_of=[1, 1, 2, 2])
        curve = learning_curve(buckets)
        # trial 1: 1/2 = 0.5, trial 2: 2/2 = 1.0 → reflexion helping
        assert curve == {1: 0.5, 2: 1.0}
        assert curve[2] > curve[1]


class TestFixtureShape:
    def test_20_trajectory_fixture_formula_ok(self):
        """DoD: 20-trajectory fixture, formulas hold."""
        trials = [
            _trial(
                success=(i % 3 != 0),  # 2/3 success rate
                steps=2 + (i % 4),
                cost=0.001 * (i + 1),
                tool_calls=(i % 5),
                tool_errors=(i % 2),
            )
            for i in range(20)
        ]
        s = summarize(trials)
        assert s.n_trials == 20
        # Manually recomputed:
        expected_success = sum(1 for i in range(20) if i % 3 != 0)
        assert s.n_success == expected_success
        assert s.success_rate == expected_success / 20
        # Tool error counts check: sum of (i%2) across non-zero (i%5) patterns
        expected_errs = sum(
            min(i % 2, i % 5) for i in range(20)
        )
        expected_total = sum(i % 5 for i in range(20))
        if expected_total:
            assert s.tool_error_rate == pytest.approx(expected_errs / expected_total)
        else:
            assert s.tool_error_rate == 0.0
