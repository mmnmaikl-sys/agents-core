"""Unit tests for agents_core.safety (tasks 0.16 confirmation + 0.17 rate_limits)."""

from __future__ import annotations

import pytest

from agents_core.safety import (
    ConfirmationRequired,
    Limit,
    RateLimitExceeded,
    RateLimits,
    require_confirmation,
)
from agents_core.tools.registry import Tool


def _tool(name: str, tier: str = "read") -> Tool:
    return Tool(
        name=name,
        description="t",
        input_schema={"type": "object", "properties": {}},
        handler=lambda **_kw: None,
        tier=tier,  # type: ignore[arg-type]
        requires_verify=(tier == "danger"),
    )


class TestConfirmation:
    def test_read_tools_never_need_confirmation(self):
        require_confirmation(_tool("deal_get"), {})  # no raise

    def test_write_tool_defaults_to_no_confirmation(self):
        require_confirmation(_tool("deal_update", "write"), {})  # no raise

    def test_write_tool_with_require_for_write_enforces(self):
        with pytest.raises(ConfirmationRequired):
            require_confirmation(
                _tool("deal_update", "write"), {}, require_for_write=True
            )

    def test_write_tool_with_require_for_write_accepts_flag(self):
        require_confirmation(
            _tool("deal_update", "write"),
            {"user_confirmed": True},
            require_for_write=True,
        )

    def test_danger_tool_blocks_without_flag(self):
        with pytest.raises(ConfirmationRequired) as ei:
            require_confirmation(_tool("deal_delete", "danger"), {})
        assert ei.value.tool_name == "deal_delete"
        assert ei.value.tier == "danger"

    def test_danger_tool_unblocks_with_flag(self):
        require_confirmation(
            _tool("deal_delete", "danger"),
            {"user_confirmed": True},
        )

    def test_custom_flag_name(self):
        require_confirmation(
            _tool("deal_delete", "danger"),
            {"double_checked": True},
            flag_name="double_checked",
        )
        with pytest.raises(ConfirmationRequired):
            require_confirmation(
                _tool("deal_delete", "danger"),
                {"user_confirmed": True},  # wrong flag for this policy
                flag_name="double_checked",
            )


class FakeClock:
    def __init__(self, t0: float = 0.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class TestRateLimits:
    def test_no_config_is_no_op(self):
        rl = RateLimits()
        for _ in range(10_000):
            rl.check_and_record(_tool("ping"))

    def test_per_tool_limit_enforces(self):
        clock = FakeClock()
        rl = RateLimits(per_tool={"hot": Limit(5, 60)}, clock=clock)
        tool = _tool("hot")
        for _ in range(5):
            rl.check_and_record(tool)
        with pytest.raises(RateLimitExceeded) as ei:
            rl.check_and_record(tool)
        assert ei.value.key == "tool:hot"
        assert ei.value.limit == 5

    def test_sliding_window_evicts(self):
        clock = FakeClock()
        rl = RateLimits(per_tool={"hot": Limit(3, 10)}, clock=clock)
        tool = _tool("hot")
        for _ in range(3):
            rl.check_and_record(tool)
        clock.advance(11)
        # Old hits evicted; 3 fresh calls allowed.
        for _ in range(3):
            rl.check_and_record(tool)
        with pytest.raises(RateLimitExceeded):
            rl.check_and_record(tool)

    def test_per_tier_limit(self):
        clock = FakeClock()
        rl = RateLimits(per_tier={"write": Limit(2, 60)}, clock=clock)
        for name in ("a", "b"):
            rl.check_and_record(_tool(name, "write"))
        with pytest.raises(RateLimitExceeded) as ei:
            rl.check_and_record(_tool("c", "write"))
        assert ei.value.key == "tier:write"

    def test_tier_and_tool_both_tracked(self):
        clock = FakeClock()
        rl = RateLimits(
            per_tool={"a": Limit(100, 60)},
            per_tier={"write": Limit(2, 60)},
            clock=clock,
        )
        rl.check_and_record(_tool("a", "write"))
        rl.check_and_record(_tool("a", "write"))
        # Tool-level fine (2 of 100), tier-level reached (2 of 2).
        with pytest.raises(RateLimitExceeded) as ei:
            rl.check_and_record(_tool("a", "write"))
        assert ei.value.key == "tier:write"

    def test_current_returns_live_count(self):
        clock = FakeClock()
        rl = RateLimits(per_tool={"a": Limit(10, 10)}, clock=clock)
        for _ in range(3):
            rl.check_and_record(_tool("a"))
        assert rl.current("tool:a") == 3
        clock.advance(11)
        assert rl.current("tool:a") == 0

    def test_current_unknown_key_zero(self):
        rl = RateLimits()
        assert rl.current("tool:nobody") == 0
