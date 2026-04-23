"""Per-tool / per-tier sliding-window rate limits (research/evaluation.md §6).

Purpose: bound blast radius. Even a well-tested tool can loop when a bug
lands in the loop/dedup fingerprint (e.g. args that encode a growing
timestamp); a rate limit is the second line of defense against run-away
token/money spend.

Sliding window: for each key we keep a deque of call timestamps, evict
entries older than ``window_seconds``, count what's left. O(1) amortised.

This module deliberately does NOT talk to Redis or a DB. Agents are
usually single-process (one Railway service = one container); when one
needs cross-process limits, wrap a Redis-backed implementation in the
same interface. `loop.dedup` uses the same pattern.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass

from agents_core.tools.registry import Tool

__all__ = ["RateLimitExceeded", "RateLimits", "Limit"]


class RateLimitExceeded(RuntimeError):  # noqa: N818  # domain term, not "Error"
    def __init__(self, key: str, limit: int, window_seconds: float) -> None:
        super().__init__(f"rate limit: {key} > {limit}/{window_seconds:.0f}s")
        self.key = key
        self.limit = limit
        self.window_seconds = window_seconds


@dataclass(frozen=True)
class Limit:
    max_calls: int
    window_seconds: float


class RateLimits:
    """Sliding-window limiter keyed by tool name and by tier.

    Tool-level limit is checked first, then tier-level (if configured). A
    call that passes both is recorded under both counters. This lets an
    operator set a tight limit on ``bitrix_deal_update`` while keeping a
    looser ceiling for all ``write`` tools combined.
    """

    def __init__(
        self,
        per_tool: dict[str, Limit] | None = None,
        per_tier: dict[str, Limit] | None = None,
        clock: callable | None = None,  # type: ignore[assignment]
    ) -> None:
        self._per_tool = dict(per_tool or {})
        self._per_tier = dict(per_tier or {})
        self._now = clock or time.monotonic
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check_and_record(self, tool: Tool) -> None:
        """Enforce both tool-level and tier-level limits, then record the hit."""
        t = self._now()
        tool_key = f"tool:{tool.name}"
        tier_key = f"tier:{tool.tier}"
        self._enforce(tool_key, self._per_tool.get(tool.name), t)
        self._enforce(tier_key, self._per_tier.get(tool.tier), t)
        # Only record if both checks passed.
        if tool.name in self._per_tool:
            self._hits[tool_key].append(t)
        if tool.tier in self._per_tier:
            self._hits[tier_key].append(t)

    def _enforce(self, key: str, limit: Limit | None, now: float) -> None:
        if limit is None:
            return
        dq = self._hits[key]
        cutoff = now - limit.window_seconds
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit.max_calls:
            raise RateLimitExceeded(key, limit.max_calls, limit.window_seconds)

    def current(self, key: str) -> int:
        """Introspection: non-evicted hit count for ``key`` right now.

        ``key`` is ``"tool:{name}"`` or ``"tier:{read|write|danger}"``.
        """
        dq = self._hits.get(key)
        if not dq:
            return 0
        # Lazy eviction so callers see consistent numbers.
        t = self._now()
        # The configured window — look up via either map.
        window = None
        if key.startswith("tool:"):
            limit = self._per_tool.get(key[5:])
            window = limit.window_seconds if limit else None
        elif key.startswith("tier:"):
            limit = self._per_tier.get(key[5:])
            window = limit.window_seconds if limit else None
        if window is not None:
            cutoff = t - window
            while dq and dq[0] < cutoff:
                dq.popleft()
        return len(dq)
