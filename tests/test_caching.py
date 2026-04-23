"""Unit tests for agents_core.llm.caching."""
from __future__ import annotations

from agents_core.llm.caching import split_input_block, wrap_system_with_cache_control


def test_split_no_input_block_returns_original():
    prompt = "Ты отличный юрист. Действуй по правилам."
    static, dynamic = split_input_block(prompt)
    assert static == ""
    assert dynamic == prompt


def test_split_with_input_block():
    prompt = "Инструкции выше.\n<input>\ntranscript: hello\n</input>\nПравила ниже."
    static, dynamic = split_input_block(prompt)
    assert "<input>" not in static
    assert "Инструкции выше" in static and "Правила ниже" in static
    assert dynamic == "<input>\ntranscript: hello\n</input>"


def test_split_multiline_input():
    prompt = """Static header.

<input>
  many
  lines
  of
  data
</input>

Static footer."""
    static, dynamic = split_input_block(prompt)
    assert "Static header" in static
    assert "Static footer" in static
    assert "<input>" in dynamic and "</input>" in dynamic
    assert "many" in dynamic and "data" in dynamic


def test_split_only_first_input_is_extracted_regex_is_non_greedy():
    prompt = "A\n<input>first</input>\nB\n<input>second</input>\nC"
    static, dynamic = split_input_block(prompt)
    # current regex strips all <input>...</input> blocks from static (sub),
    # but dynamic keeps only the first match
    assert "<input>" not in static
    assert dynamic == "<input>first</input>"


def test_wrap_system_with_cache_control_structure():
    out = wrap_system_with_cache_control("Long stable system prompt")
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["type"] == "text"
    assert out[0]["text"] == "Long stable system prompt"
    assert out[0]["cache_control"] == {"type": "ephemeral"}
