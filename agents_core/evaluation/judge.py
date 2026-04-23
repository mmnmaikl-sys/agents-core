"""Multi-signal evaluator (research/evaluation.md §3 "Хороший evaluator").

One LLM-judge can systematically rubber-stamp "yes" — the research explicitly
warns against this. ``Judge`` combines structural, statistical, and optional
LLM signals so any single signal going wrong does not make the whole run
"successful".

Signals (each returns ``True`` / ``False`` / ``None``; ``None`` = not
evaluated, skipped in aggregation):

1. ``finish``           stop_reason == "end_turn"
2. ``tool_error_rate``  errors / total_calls <= threshold
3. ``schema``           final_text validates against the given pydantic model
4. ``heuristic``        no "не знаю"-style dodge when ground_truth was promised
5. ``llm_judge``        Haiku rates 1..5, >= min_rating

Aggregation: allow at most 1 failing signal (matches research code:
``sum(v) >= len(signals) - 1``). This is deliberate — strict-AND would make
the judge brittle and penalise legitimate partial answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agents_core.llm.client import LLMClient
from agents_core.loop.react import LoopResult

__all__ = ["JudgeResult", "Judge"]


@dataclass
class JudgeResult:
    success: bool
    signals: dict[str, bool | None]
    reason: str
    llm_rating: int | None = None

    @property
    def failed_signals(self) -> list[str]:
        return [k for k, v in self.signals.items() if v is False]


class _LLMJudgeVerdict(BaseModel):
    """Structured output for the optional LLM judge step."""

    rating: int = Field(ge=1, le=5)
    reasoning: str


class Judge:
    """Pluggable evaluator. Defaults are safe; override per-agent if needed.

    Parameters
    ----------
    client:
        Optional ``LLMClient``. If omitted, the ``llm_judge`` signal is
        skipped — useful for unit tests and heuristic-only pipelines.
    judge_model:
        Model id for the LLM judge. Research §3 recommends Haiku.
    tool_error_threshold:
        Max error fraction; default 0.30 matches research §3.
    llm_min_rating:
        Lowest LLM rating (1..5) that counts as success. Default 4.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        judge_model: str = "haiku",
        tool_error_threshold: float = 0.30,
        llm_min_rating: int = 4,
    ) -> None:
        if not 0.0 <= tool_error_threshold <= 1.0:
            raise ValueError("tool_error_threshold must be in [0, 1]")
        if not 1 <= llm_min_rating <= 5:
            raise ValueError("llm_min_rating must be in [1, 5]")
        self._client = client
        self._judge_model = judge_model
        self._error_threshold = tool_error_threshold
        self._min_rating = llm_min_rating

    @staticmethod
    def tool_error_rate(trajectory: LoopResult) -> tuple[int, int]:
        """Return (errors, total) across all tool_result blocks."""
        errors = 0
        total = 0
        for step in trajectory.steps:
            for tr in step.tool_results:
                total += 1
                if tr.get("is_error"):
                    errors += 1
        return errors, total

    async def evaluate(
        self,
        task: str,
        trajectory: LoopResult,
        *,
        schema: type[BaseModel] | None = None,
        ground_truth: Any | None = None,
        heuristic_dodge_phrases: tuple[str, ...] = (
            "не знаю",
            "не нашёл",
            "не нашел",
            "cannot help",
        ),
    ) -> JudgeResult:
        signals: dict[str, bool | None] = {}

        # 1. structural finish
        signals["finish"] = trajectory.stop_reason == "end_turn"

        # 2. tool error rate
        errs, total = self.tool_error_rate(trajectory)
        if total == 0:
            signals["tool_error_rate"] = None  # no tools used → skip
        else:
            signals["tool_error_rate"] = (errs / total) <= self._error_threshold

        # 3. schema
        if schema is not None:
            try:
                schema.model_validate_json(trajectory.final_text)
                signals["schema"] = True
            except (ValidationError, ValueError):
                signals["schema"] = False
        else:
            signals["schema"] = None

        # 4. heuristic "gave up" only when ground_truth was promised
        if ground_truth is not None:
            text = trajectory.final_text.lower()
            dodged = any(p in text for p in heuristic_dodge_phrases)
            signals["heuristic"] = not dodged
        else:
            signals["heuristic"] = None

        # 5. LLM judge (optional)
        llm_rating: int | None = None
        signals["llm_judge"] = None
        if self._client is not None:
            try:
                verdict, _ = await self._client.chat_structured(
                    prompt=(
                        f"Task:\n{task}\n\n"
                        f"Agent's final answer:\n{trajectory.final_text}\n\n"
                        "Rate how well the answer solves the task on a 1..5 scale."
                    ),
                    response_model=_LLMJudgeVerdict,
                    model=self._judge_model,
                    system=(
                        "You are a strict evaluator. Reply with JSON {rating, "
                        "reasoning}. Be skeptical: mark 3 or below if the "
                        "answer is vague, refuses, or hallucinates."
                    ),
                    max_tokens=256,
                    name="judge.llm_verdict",
                )
                llm_rating = verdict.rating
                signals["llm_judge"] = llm_rating >= self._min_rating
            except Exception:
                signals["llm_judge"] = None  # judge unavailable → skip signal

        # Aggregation: `finish` is mandatory (hitting max_tokens / mid-loop
        # stop is never "success"). Among the rest of the active signals,
        # allow at most one failure (research §3 formula).
        finish_ok = signals["finish"] is True
        others = [
            (k, v)
            for k, v in signals.items()
            if k != "finish" and v is not None
        ]
        other_passes = sum(1 for _, v in others if v)
        others_ok = other_passes >= len(others) - 1  # True also when others=[]
        success = finish_ok and others_ok

        all_active = [v for v in signals.values() if v is not None]
        total_passes = sum(1 for v in all_active if v)
        reason = (
            "all signals passed"
            if success and total_passes == len(all_active)
            else f"{total_passes}/{len(all_active)} signals passed"
        )
        return JudgeResult(
            success=success,
            signals=signals,
            reason=reason,
            llm_rating=llm_rating,
        )
