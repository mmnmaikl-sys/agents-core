"""Complexity-based model routing.

Uses Haiku as a cheap classifier to decide whether a task should go to
haiku / sonnet / opus. The calling agent saves money by sending short/routine
queries to Haiku and reserving Sonnet/Opus for genuinely hard tasks.

Three tiers:
    haiku  — simple classification, lookup, formatting, short summaries (< 2 steps)
    sonnet — multi-step reasoning, structured extraction, most agent tool-loops
    opus   — complex planning, deep analysis, novel/legal nuance (rare)

Usage:
    router = ComplexityRouter(llm_client)
    tier = await router.classify("Что делать с недовольным клиентом?")
    # → "sonnet"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agents_core.llm.client import LLMClient

Tier = Literal["haiku", "sonnet", "opus"]

_VALID_TIERS: tuple[Tier, ...] = ("haiku", "sonnet", "opus")

_CLASSIFIER_SYSTEM = (
    "Ты классификатор сложности задач для LLM-маршрутизации. "
    "На вход — задача. На выход — строго ОДНО слово: haiku, sonnet или opus.\n\n"
    "haiku — простая классификация, извлечение факта, короткий ответ, форматирование, "
    "routine lookup. Решается за ≤2 шага рассуждения.\n"
    "sonnet — многошаговое рассуждение, структурное извлечение, tool-use loop с 3-7 "
    "шагами, средняя юр/финансовая нюансировка.\n"
    "opus — глубокое планирование, сложная юр/финансовая нюансировка, многомерные "
    "trade-offs, редкие edge cases. Используй редко.\n\n"
    "Отвечай ТОЛЬКО одним словом из списка: haiku, sonnet, opus."
)


@dataclass
class RoutingDecision:
    tier: Tier
    raw_reply: str
    cost_usd: float
    input_tokens: int
    output_tokens: int


class ComplexityRouter:
    """Classifies a task into haiku / sonnet / opus using a Haiku call."""

    def __init__(self, llm: LLMClient, fallback: Tier = "sonnet") -> None:
        self._llm = llm
        self._fallback = fallback

    async def classify(self, task: str, name: str = "llm.routing") -> Tier:
        decision = await self.classify_detailed(task, name=name)
        return decision.tier

    async def classify_detailed(self, task: str, name: str = "llm.routing") -> RoutingDecision:
        resp = await self._llm.chat(
            prompt=f"Задача:\n{task}\n\nОтветь одним словом: haiku, sonnet или opus.",
            model="haiku",
            system=_CLASSIFIER_SYSTEM,
            max_tokens=8,
            name=name,
        )
        tier = _parse_tier(resp.text, self._fallback)
        return RoutingDecision(
            tier=tier,
            raw_reply=resp.text,
            cost_usd=resp.cost_usd,
            input_tokens=resp.usage.input,
            output_tokens=resp.usage.output,
        )


def _parse_tier(text: str, fallback: Tier) -> Tier:
    """Extract tier from free-form reply. Robust to quoting, punctuation, and case."""
    token = text.strip().lower().strip(".,!?\"'` ")
    # model sometimes produces leading filler — take first valid token
    for word in token.replace("\n", " ").split():
        w = word.strip(".,!?\"'`-:")
        if w in _VALID_TIERS:
            return w  # type: ignore[return-value]
    return fallback
