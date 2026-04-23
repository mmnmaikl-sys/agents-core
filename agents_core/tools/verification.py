"""SET/VERIFY pair runner — symmetry of action enforced (research §11 principle 2).

Write / danger tools must be followed by a VERIFY tool that reads the post-state
and confirms the expected change actually happened. This module runs both calls
through a `ToolRegistry` and asserts that the VERIFY result contains the expected
fields with the expected values. On mismatch it either returns a structured
`VerificationResult(verified=False, mismatches=[...])` or raises
`VerificationError` (default — `raise_on_mismatch=True`).

Used by `loop/react.py` right after every tool_use with tier ∈ {write, danger}.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agents_core.tools.registry import ToolRegistry

Matcher = Callable[[Any, Any], bool]


def default_verify_name(set_tool_name: str) -> str:
    """Convention: deal_update → deal_verify, deal_close_lost → deal_close_lost_verify.

    If the name ends with a known SET verb (`_update`, `_insert`, `_create`,
    `_delete`, `_set`, `_add`), swap the verb for `_verify`; otherwise append
    `_verify`. The returned name is a suggestion — callers can override.
    """
    suffixes = ("_update", "_insert", "_create", "_delete", "_set", "_add")
    for s in suffixes:
        if set_tool_name.endswith(s):
            return set_tool_name[: -len(s)] + "_verify"
    return set_tool_name + "_verify"


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    set_result: Any
    verify_result: Any
    mismatches: list[tuple[str, Any, Any]] = field(default_factory=list)

    def summary(self) -> str:
        if self.verified:
            return "verified"
        parts = [f"{k}: expected={e!r} actual={a!r}" for k, e, a in self.mismatches]
        return "mismatch — " + "; ".join(parts)


class VerificationError(RuntimeError):
    def __init__(self, result: VerificationResult) -> None:
        super().__init__(result.summary())
        self.result = result


def _equal(expected: Any, actual: Any) -> bool:
    return expected == actual


def _extract(obj: Any, key: str) -> Any:
    """Dig a value by key from dict or object attribute. Returns `_MISSING` if absent."""
    if isinstance(obj, Mapping):
        return obj.get(key, _MISSING)
    return getattr(obj, key, _MISSING)


_MISSING = object()


async def run_with_verify(
    registry: ToolRegistry,
    *,
    set_tool: str,
    set_args: dict[str, Any],
    verify_tool: str,
    verify_args: dict[str, Any],
    expected: dict[str, Any],
    matcher: Matcher = _equal,
    raise_on_mismatch: bool = True,
) -> VerificationResult:
    """Run SET tool then VERIFY tool, compare expected ⊆ verify_result.

    Returns VerificationResult. Raises VerificationError when
    raise_on_mismatch=True (default) and verification fails.

    Matcher signature: (expected_value, actual_value) -> bool. Default is
    strict equality. Pass a fuzzy matcher if the verify tool returns a
    superset or a transformed value.
    """
    set_result = await registry.call(set_tool, **set_args)
    verify_result = await registry.call(verify_tool, **verify_args)

    mismatches: list[tuple[str, Any, Any]] = []
    for key, exp_val in expected.items():
        actual = _extract(verify_result, key)
        if actual is _MISSING or not matcher(exp_val, actual):
            mismatches.append((key, exp_val, None if actual is _MISSING else actual))

    result = VerificationResult(
        verified=not mismatches,
        set_result=set_result,
        verify_result=verify_result,
        mismatches=mismatches,
    )

    if not result.verified and raise_on_mismatch:
        raise VerificationError(result)
    return result
