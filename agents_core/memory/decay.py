"""Age-based scoring for ReflectionStore retrieval (research §3 layer 4).

Plug as ``score_fn`` into ``ReflectionStore.retrieve``:

    from agents_core.memory.decay import age_decay_score
    top3 = await store.retrieve(h, k=3, score_fn=age_decay_score)

Score is ``confidence × exp(-age_days / half_life_days)``. Older reflections
rank lower; high-confidence new ones win.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime

from .reflection import Reflection

__all__ = ["age_decay_score", "make_age_decay_score"]


def make_age_decay_score(
    half_life_days: float = 30.0,
    now: datetime | None = None,
) -> Callable[[Reflection], float]:
    """Return a score_fn with a configurable half-life.

    ``half_life_days=30`` matches research §3: "старше 30 дней → ниже
    приоритет".
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be > 0")
    t_now = now or datetime.now(UTC)
    tau = half_life_days * 86400.0  # seconds

    def _score(r: Reflection) -> float:
        age_s = r.age_seconds(t_now)
        return r.confidence * math.exp(-age_s / tau * math.log(2))

    return _score


age_decay_score: Callable[[Reflection], float] = make_age_decay_score()
