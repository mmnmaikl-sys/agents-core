"""Unit tests for agents_core.tools.verification.

DoD: simulate SET → VERIFY flow, assert raise on mismatch.

Covers:
- successful SET → VERIFY with expected fields present → verified=True
- mismatched VERIFY → VerificationError raised by default
- raise_on_mismatch=False → returns result with mismatches list
- missing key in verify_result → treated as mismatch
- default_verify_name convention mapping
- custom matcher (e.g., case-insensitive string compare)
"""
from __future__ import annotations

import pytest

from agents_core.tools import (
    Tool,
    ToolRegistry,
    VerificationError,
    default_verify_name,
    run_with_verify,
)


# ---------------------------------------------------------- fixtures
def _build_deal_registry(verify_result: dict) -> ToolRegistry:
    """Registry with a fake deal_update + deal_verify pair."""
    updated: dict = {}

    def deal_update(deal_id, fields, idempotency_key):
        updated["deal_id"] = deal_id
        updated["fields"] = fields
        return {"ok": True, "deal_id": deal_id}

    def deal_verify(deal_id):
        return verify_result

    r = ToolRegistry()
    r.register(Tool(
        name="deal_update",
        description="Update deal",
        input_schema={
            "type": "object",
            "properties": {
                "deal_id": {"type": "string"},
                "fields": {"type": "object"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["deal_id", "fields", "idempotency_key"],
        },
        handler=deal_update,
        tier="write",
        requires_verify=True,
    ))
    r.register(Tool(
        name="deal_verify",
        description="Fetch deal state for verification",
        input_schema={
            "type": "object",
            "properties": {"deal_id": {"type": "string"}},
            "required": ["deal_id"],
        },
        handler=deal_verify,
    ))
    return r


# ---------------------------------------------------------- default_verify_name
@pytest.mark.parametrize("src,exp", [
    ("deal_update", "deal_verify"),
    ("contact_insert", "contact_verify"),
    ("task_create", "task_verify"),
    ("user_delete", "user_verify"),
    ("stage_set", "stage_verify"),
    ("activity_add", "activity_verify"),
    ("something_weird", "something_weird_verify"),  # no known suffix → append
])
def test_default_verify_name_convention(src: str, exp: str):
    assert default_verify_name(src) == exp


# ---------------------------------------------------------- happy path
@pytest.mark.asyncio
async def test_verify_success():
    r = _build_deal_registry({"stage_id": "C45:WON", "amount": 150000})
    result = await run_with_verify(
        r,
        set_tool="deal_update",
        set_args={
            "deal_id": "42",
            "fields": {"stage_id": "C45:WON"},
            "idempotency_key": "abc-123",
        },
        verify_tool="deal_verify",
        verify_args={"deal_id": "42"},
        expected={"stage_id": "C45:WON"},
    )
    assert result.verified is True
    assert result.mismatches == []
    assert result.set_result == {"ok": True, "deal_id": "42"}
    assert result.verify_result["stage_id"] == "C45:WON"
    assert result.summary() == "verified"


# ---------------------------------------------------------- mismatch raises
@pytest.mark.asyncio
async def test_verify_mismatch_raises():
    r = _build_deal_registry({"stage_id": "C45:LOST"})  # SET failed silently
    with pytest.raises(VerificationError) as exc:
        await run_with_verify(
            r,
            set_tool="deal_update",
            set_args={"deal_id": "42", "fields": {"stage_id": "C45:WON"}, "idempotency_key": "k"},
            verify_tool="deal_verify",
            verify_args={"deal_id": "42"},
            expected={"stage_id": "C45:WON"},
        )
    assert exc.value.result.verified is False
    assert exc.value.result.mismatches == [("stage_id", "C45:WON", "C45:LOST")]
    assert "stage_id" in str(exc.value)


@pytest.mark.asyncio
async def test_verify_mismatch_no_raise():
    r = _build_deal_registry({"stage_id": "C45:LOST"})
    result = await run_with_verify(
        r,
        set_tool="deal_update",
        set_args={"deal_id": "42", "fields": {"stage_id": "C45:WON"}, "idempotency_key": "k"},
        verify_tool="deal_verify",
        verify_args={"deal_id": "42"},
        expected={"stage_id": "C45:WON"},
        raise_on_mismatch=False,
    )
    assert result.verified is False
    assert len(result.mismatches) == 1
    assert "mismatch" in result.summary()


# ---------------------------------------------------------- missing key
@pytest.mark.asyncio
async def test_verify_missing_key_is_mismatch():
    # verify tool returns a dict that lacks the key we expect
    r = _build_deal_registry({"unrelated_field": 1})
    with pytest.raises(VerificationError) as exc:
        await run_with_verify(
            r,
            set_tool="deal_update",
            set_args={"deal_id": "42", "fields": {"stage_id": "C45:WON"}, "idempotency_key": "k"},
            verify_tool="deal_verify",
            verify_args={"deal_id": "42"},
            expected={"stage_id": "C45:WON"},
        )
    # missing key → actual reported as None in mismatch tuple
    assert exc.value.result.mismatches == [("stage_id", "C45:WON", None)]


# ---------------------------------------------------------- multiple fields
@pytest.mark.asyncio
async def test_verify_multiple_fields_partial_mismatch():
    r = _build_deal_registry({"stage_id": "C45:WON", "amount": 100000})
    with pytest.raises(VerificationError) as exc:
        await run_with_verify(
            r,
            set_tool="deal_update",
            set_args={"deal_id": "42", "fields": {}, "idempotency_key": "k"},
            verify_tool="deal_verify",
            verify_args={"deal_id": "42"},
            expected={"stage_id": "C45:WON", "amount": 150000},
        )
    # stage matches, amount doesn't → exactly 1 mismatch
    assert exc.value.result.mismatches == [("amount", 150000, 100000)]


# ---------------------------------------------------------- custom matcher
@pytest.mark.asyncio
async def test_verify_custom_matcher_case_insensitive():
    r = _build_deal_registry({"stage_id": "c45:won"})

    def ci_match(expected, actual):
        if isinstance(expected, str) and isinstance(actual, str):
            return expected.lower() == actual.lower()
        return expected == actual

    result = await run_with_verify(
        r,
        set_tool="deal_update",
        set_args={"deal_id": "42", "fields": {}, "idempotency_key": "k"},
        verify_tool="deal_verify",
        verify_args={"deal_id": "42"},
        expected={"stage_id": "C45:WON"},
        matcher=ci_match,
    )
    assert result.verified is True


# ---------------------------------------------------------- extract from object
@pytest.mark.asyncio
async def test_verify_with_object_return():
    from types import SimpleNamespace

    def set_tool_fn():
        return SimpleNamespace(ok=True)

    def verify_tool_fn(deal_id):
        return SimpleNamespace(stage_id="C45:WON", amount=150000)

    r = ToolRegistry()
    r.register(Tool(
        name="noop_set",
        description="Noop setter returning an object",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=set_tool_fn,
    ))
    r.register(Tool(
        name="obj_verify",
        description="Verify returning an object (not a dict)",
        input_schema={
            "type": "object",
            "properties": {"deal_id": {"type": "string"}},
            "required": ["deal_id"],
        },
        handler=verify_tool_fn,
    ))

    result = await run_with_verify(
        r,
        set_tool="noop_set",
        set_args={},
        verify_tool="obj_verify",
        verify_args={"deal_id": "42"},
        expected={"stage_id": "C45:WON"},
    )
    assert result.verified is True
