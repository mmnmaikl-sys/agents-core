"""Unit tests for agents_core.templates (task 0.23)."""

from __future__ import annotations

import pytest

from agents_core.templates import CANONICAL_SYSTEM_PROMPT, load_template


def test_canonical_loaded_as_constant():
    assert isinstance(CANONICAL_SYSTEM_PROMPT, str)
    assert CANONICAL_SYSTEM_PROMPT.strip(), "template must not be empty"


def test_canonical_contains_all_ten_rules():
    """Research §9 has 9 numbered rules. Every ordinal must be present."""
    for i in range(1, 10):
        assert f"\n{i}. " in CANONICAL_SYSTEM_PROMPT, f"rule {i} missing"


def test_canonical_contains_stop_conditions():
    for marker in ("СТОП-УСЛОВИЯ", "Max 12 steps", "Budget $0.50"):
        assert marker in CANONICAL_SYSTEM_PROMPT, f"missing: {marker!r}"


def test_canonical_mentions_set_verify_symmetry():
    assert "_verify" in CANONICAL_SYSTEM_PROMPT
    assert "verified=true" in CANONICAL_SYSTEM_PROMPT


def test_load_template_unknown_raises():
    with pytest.raises(FileNotFoundError):
        load_template("definitely-not-a-template")


def test_load_template_roundtrip():
    assert load_template("canonical") == CANONICAL_SYSTEM_PROMPT
