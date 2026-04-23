"""Unit tests for agents_core.evaluation.judge (task 0.12)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from agents_core.evaluation import Judge
from agents_core.evaluation.judge import _LLMJudgeVerdict
from agents_core.llm.client import AgentTurn, LLMResponse, LLMUsage
from agents_core.loop.react import LoopResult, StepRecord


def _turn(stop_reason: str = "end_turn", text: str = "ok") -> AgentTurn:
    return AgentTurn(
        model="claude-haiku-4-5-20251001",
        stop_reason=stop_reason,
        text=text,
        tool_uses=[],
        message_content=[{"type": "text", "text": text}],
        usage=LLMUsage(input=10, output=5),
        cost_usd=0.0001,
        duration_sec=0.5,
    )


def _traj(
    *,
    final: str = "done",
    stop_reason: str = "end_turn",
    tool_results: list[list[dict]] | None = None,
) -> LoopResult:
    """Build a trajectory with per-step tool_results (each inner list = one step)."""
    steps: list[StepRecord] = []
    tool_results = tool_results or [[]]
    for i, tr in enumerate(tool_results, start=1):
        steps.append(StepRecord(step=i, turn=_turn(), tool_results=list(tr)))
    return LoopResult(
        final_text=final,
        stop_reason=stop_reason,
        steps=steps,
        messages=[],
        usage=LLMUsage(input=10, output=5),
        cost_usd=0.0001,
    )


class TestJudgeValidation:
    def test_error_threshold_out_of_range(self):
        with pytest.raises(ValueError):
            Judge(tool_error_threshold=1.5)
        with pytest.raises(ValueError):
            Judge(tool_error_threshold=-0.1)

    def test_min_rating_out_of_range(self):
        with pytest.raises(ValueError):
            Judge(llm_min_rating=0)
        with pytest.raises(ValueError):
            Judge(llm_min_rating=6)


class TestToolErrorRate:
    def test_counts_errors_and_total(self):
        traj = _traj(
            tool_results=[
                [{"type": "tool_result", "is_error": False}],
                [
                    {"type": "tool_result", "is_error": True},
                    {"type": "tool_result", "is_error": False},
                ],
            ]
        )
        errors, total = Judge.tool_error_rate(traj)
        assert (errors, total) == (1, 3)

    def test_empty_trajectory_zero_zero(self):
        traj = _traj(tool_results=[[]])
        assert Judge.tool_error_rate(traj) == (0, 0)


class TestEvaluateHeuristicsOnly:
    @pytest.mark.asyncio
    async def test_all_pass_without_tools_without_llm(self):
        judge = Judge()  # no client
        result = await judge.evaluate("task", _traj(final="clear answer"))
        assert result.success is True
        assert result.signals["finish"] is True
        assert result.signals["tool_error_rate"] is None  # no tools
        assert result.signals["schema"] is None
        assert result.signals["heuristic"] is None
        assert result.signals["llm_judge"] is None

    @pytest.mark.asyncio
    async def test_unfinished_trajectory_fails_finish(self):
        judge = Judge()
        result = await judge.evaluate("task", _traj(stop_reason="max_tokens"))
        assert result.success is False
        assert result.signals["finish"] is False

    @pytest.mark.asyncio
    async def test_tool_error_over_threshold_fails(self):
        judge = Judge(tool_error_threshold=0.3)
        traj = _traj(
            tool_results=[
                [
                    {"is_error": True},
                    {"is_error": True},
                    {"is_error": False},
                ]
            ]
        )
        result = await judge.evaluate("task", traj)
        # 2/3 > 0.3 → tool_error_rate False; finish True → 1 failing allowed
        assert result.signals["tool_error_rate"] is False
        # Only one failing signal and one passing — len-1 rule says success.
        assert result.success is True
        # Add another failing signal to flip it:
        judge2 = Judge()
        traj_bad_finish = _traj(
            stop_reason="max_tokens",
            tool_results=[[{"is_error": True}, {"is_error": True}]],
        )
        result2 = await judge2.evaluate("task", traj_bad_finish)
        assert result2.success is False

    @pytest.mark.asyncio
    async def test_heuristic_gave_up_when_ground_truth_expected(self):
        judge = Judge()
        traj = _traj(final="Не знаю — данных нет")
        result = await judge.evaluate("task", traj, ground_truth={"x": 1})
        assert result.signals["heuristic"] is False

    @pytest.mark.asyncio
    async def test_heuristic_skipped_without_ground_truth(self):
        judge = Judge()
        traj = _traj(final="Не знаю")
        result = await judge.evaluate("task", traj)  # no ground_truth
        assert result.signals["heuristic"] is None

    @pytest.mark.asyncio
    async def test_schema_validation_pass(self):
        class Answer(BaseModel):
            value: int

        judge = Judge()
        traj = _traj(final='{"value": 7}')
        result = await judge.evaluate("t", traj, schema=Answer)
        assert result.signals["schema"] is True

    @pytest.mark.asyncio
    async def test_schema_validation_fail(self):
        class Answer(BaseModel):
            value: int

        judge = Judge()
        traj = _traj(final='not json at all')
        result = await judge.evaluate("t", traj, schema=Answer)
        assert result.signals["schema"] is False


class TestEvaluateWithLLMJudge:
    @pytest.mark.asyncio
    async def test_llm_judge_pass(self):
        client = AsyncMock()
        client.chat_structured.return_value = (
            _LLMJudgeVerdict(rating=5, reasoning="great"),
            LLMResponse(text="", model="haiku"),
        )
        judge = Judge(client=client, llm_min_rating=4)
        result = await judge.evaluate("t", _traj(final="ok"))
        assert result.signals["llm_judge"] is True
        assert result.llm_rating == 5

    @pytest.mark.asyncio
    async def test_llm_judge_fail(self):
        client = AsyncMock()
        client.chat_structured.return_value = (
            _LLMJudgeVerdict(rating=2, reasoning="vague"),
            LLMResponse(text="", model="haiku"),
        )
        judge = Judge(client=client, llm_min_rating=4)
        result = await judge.evaluate("t", _traj(final="dunno"))
        assert result.signals["llm_judge"] is False
        assert result.llm_rating == 2

    @pytest.mark.asyncio
    async def test_llm_judge_exception_is_skipped(self):
        client = AsyncMock()
        client.chat_structured.side_effect = RuntimeError("api down")
        judge = Judge(client=client)
        result = await judge.evaluate("t", _traj(final="ok"))
        assert result.signals["llm_judge"] is None
        # finish is True, llm skipped → result.success depends on other signals
        assert result.success is True  # finish alone passes

    @pytest.mark.asyncio
    async def test_five_trajectories_end_to_end(self):
        """DoD: 5 trajectories, judge returns success=True/False with reasons."""
        judge = Judge()
        cases = [
            (_traj(final="good", stop_reason="end_turn"), True),
            (_traj(final="bad", stop_reason="max_tokens"), False),
            (
                _traj(
                    final="ok",
                    tool_results=[[{"is_error": True}, {"is_error": True}]],
                ),
                True,  # 1 failing signal allowed
            ),
            (_traj(final="Не нашёл ничего"), True),  # no ground_truth → skipped
            (_traj(final="{}", stop_reason="end_turn"), True),
        ]
        for traj, want in cases:
            r = await judge.evaluate("task", traj)
            assert r.success == want, (traj.final_text, r.signals, r.reason)
            assert r.reason  # non-empty
