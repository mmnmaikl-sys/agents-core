"""Unit tests for agents_core.loop.dedup.LoopDetector.

DoD: 3 identical tool calls in a row → LoopDetectedError raised.

Covers:
- 3rd identical call triggers (default threshold=3)
- 2 identical + 1 different → no trigger
- Different args with same tool name don't count together
- args order-independent (dict sort_keys in fingerprint)
- non-serializable args fall back to str()
- reset() clears window
- threshold validation
"""
from __future__ import annotations

import pytest

from agents_core.loop.dedup import LoopDetectedError, LoopDetector


# ---------------------------------------------------------- DoD
def test_three_identical_calls_trigger():
    d = LoopDetector(threshold=3)
    d.observe("deal_update", {"deal_id": "42"})
    d.observe("deal_update", {"deal_id": "42"})
    with pytest.raises(LoopDetectedError) as exc:
        d.observe("deal_update", {"deal_id": "42"})
    assert exc.value.repeat.tool == "deal_update"
    assert exc.value.repeat.count == 3
    assert exc.value.repeat.threshold == 3


def test_two_identical_plus_different_no_trigger():
    d = LoopDetector(threshold=3)
    d.observe("deal_update", {"deal_id": "42"})
    d.observe("deal_update", {"deal_id": "42"})
    d.observe("deal_update", {"deal_id": "99"})  # different args
    # no raise


def test_different_tools_same_args_no_trigger():
    d = LoopDetector(threshold=3)
    d.observe("deal_get", {"deal_id": "42"})
    d.observe("deal_update", {"deal_id": "42"})
    d.observe("deal_verify", {"deal_id": "42"})
    # same args, different tools — not a loop


# ---------------------------------------------------------- fingerprint stability
def test_args_order_independent():
    d = LoopDetector(threshold=2)
    d.observe("t", {"a": 1, "b": 2})
    with pytest.raises(LoopDetectedError):
        d.observe("t", {"b": 2, "a": 1})  # same after sort_keys


def test_nested_args_fingerprinted():
    d = LoopDetector(threshold=2)
    d.observe("t", {"list": [1, 2, 3], "obj": {"x": 1}})
    with pytest.raises(LoopDetectedError):
        d.observe("t", {"list": [1, 2, 3], "obj": {"x": 1}})


def test_non_serializable_args_fallback():
    d = LoopDetector(threshold=2)

    class Obj:
        def __repr__(self) -> str:
            return "Obj()"

    # Should not crash — default=str converts
    d.observe("t", {"x": Obj()})
    with pytest.raises(LoopDetectedError):
        d.observe("t", {"x": Obj()})


def test_none_args_treated_as_empty_dict():
    d = LoopDetector(threshold=2)
    d.observe("t", None)
    # both None and {} normalized to same fingerprint — 2nd triggers
    with pytest.raises(LoopDetectedError):
        d.observe("t", {})


# ---------------------------------------------------------- window
def test_window_evicts_old_calls():
    # threshold=3, window=3 — only last 3 calls counted
    d = LoopDetector(threshold=3, window=3)
    d.observe("t", {"a": 1})
    d.observe("t", {"a": 1})
    d.observe("t", {"b": 2})  # breaks streak, evicts oldest on next
    d.observe("t", {"b": 2})
    # now window holds: a, b, b — only 2 of the same → still ok
    d.observe("t", {"a": 1})  # pushes out first {"a":1}, window: b, b, a → ok
    # Add 2 more "a"s — then a count reaches 3
    with pytest.raises(LoopDetectedError):
        d.observe("t", {"a": 1})  # window now b, a, a — no
        d.observe("t", {"a": 1})  # window a, a, a → trigger


def test_default_window_is_double_threshold():
    d = LoopDetector(threshold=4)
    assert d.window == 8
    assert d.threshold == 4


# ---------------------------------------------------------- reset
def test_reset_clears_history():
    d = LoopDetector(threshold=2)
    d.observe("t", {})
    d.reset()
    # after reset, first observe should NOT trigger (count == 1)
    d.observe("t", {})
    # second one does
    with pytest.raises(LoopDetectedError):
        d.observe("t", {})


# ---------------------------------------------------------- validation
def test_threshold_min_2():
    with pytest.raises(ValueError, match="threshold must be >= 2"):
        LoopDetector(threshold=1)


def test_error_message_carries_tool_and_count():
    d = LoopDetector(threshold=2)
    d.observe("bitrix_deal_update", {"deal_id": "abc"})
    with pytest.raises(LoopDetectedError) as exc:
        d.observe("bitrix_deal_update", {"deal_id": "abc"})
    msg = str(exc.value)
    assert "bitrix_deal_update" in msg
    assert "2 times" in msg
