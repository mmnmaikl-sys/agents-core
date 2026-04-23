"""Unit tests for agents_core.loop.react.

Mocks LLMClient.chat_with_tools to drive the loop through fake turns:
- tool_use → tool_result → end_turn (DoD: hello-world with 3 tools, 1 cycle)
- parallel tool use (two tool_use blocks in one turn)
- tool raising → is_error=true result, model continues
- max_steps reached → MaxStepsExceededError
- max_tokens stop_reason → early return with partial text
- single turn end_turn without any tools
- message history correctness (assistant + user tool_results appended)
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents_core.llm.client import AgentTurn, LLMUsage, ToolUse
from agents_core.loop import LoopResult, MaxStepsExceededError, ReActLoop
from agents_core.tools import Tool, ToolRegistry


# ---------------------------------------------------------- helpers
def _turn(stop: str, text: str = "", tool_uses: list[ToolUse] | None = None) -> AgentTurn:
    tool_uses = tool_uses or []
    content: list[Any] = []
    if text:
        content.append({"type": "text", "text": text})
    for tu in tool_uses:
        content.append({
            "type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input,
        })
    return AgentTurn(
        model="claude-sonnet-4-20250514",
        stop_reason=stop,
        text=text,
        tool_uses=tool_uses,
        message_content=content,
        usage=LLMUsage(input=10, output=5),
        cost_usd=0.00005,
    )


def _mini_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(Tool(
        name="get_time",
        description="Return current time as ISO string",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=lambda: "2026-04-23T10:00:00Z",
    ))
    r.register(Tool(
        name="add",
        description="Add two numbers",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        handler=lambda a, b: a + b,
    ))
    r.register(Tool(
        name="fail",
        description="Always raises",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    ))
    return r


def _mock_client(turn_sequence: list[AgentTurn]) -> MagicMock:
    client = MagicMock()
    client.chat_with_tools = AsyncMock(side_effect=turn_sequence)
    return client


# ---------------------------------------------------------- DoD: hello-world 3 tools, 1 cycle
@pytest.mark.asyncio
async def test_hello_world_tool_use_then_end_turn():
    """Canonical path: 1 tool_use turn → tool_result → 1 end_turn."""
    registry = _mini_registry()
    client = _mock_client([
        _turn("tool_use", tool_uses=[
            ToolUse(id="tu_1", name="get_time", input={}),
        ]),
        _turn("end_turn", text="The time is 2026-04-23T10:00:00Z"),
    ])

    loop = ReActLoop(client, registry, system="You are a test agent.", model="sonnet")
    result = await loop.run(task="What time is it?")

    assert isinstance(result, LoopResult)
    assert result.final_text == "The time is 2026-04-23T10:00:00Z"
    assert result.stop_reason == "end_turn"
    assert result.step_count == 2

    # Step 1: one tool_use → one tool_result
    step1 = result.steps[0]
    assert len(step1.turn.tool_uses) == 1
    assert step1.tool_results == [{
        "type": "tool_result",
        "tool_use_id": "tu_1",
        "content": "2026-04-23T10:00:00Z",
        "is_error": False,
    }]

    # Step 2: no tool_use, end_turn
    assert result.steps[1].turn.stop_reason == "end_turn"
    assert result.steps[1].tool_results == []


# ---------------------------------------------------------- parallel tool use
@pytest.mark.asyncio
async def test_parallel_tool_use_two_tools_one_turn():
    registry = _mini_registry()
    client = _mock_client([
        _turn("tool_use", tool_uses=[
            ToolUse(id="tu_1", name="get_time", input={}),
            ToolUse(id="tu_2", name="add", input={"a": 3, "b": 4}),
        ]),
        _turn("end_turn", text="done"),
    ])
    loop = ReActLoop(client, registry, system="s")
    result = await loop.run(task="do it")

    assert result.final_text == "done"
    tr = result.steps[0].tool_results
    assert len(tr) == 2
    # order matters — must match tool_use order
    assert tr[0]["tool_use_id"] == "tu_1" and tr[0]["content"] == "2026-04-23T10:00:00Z"
    assert tr[1]["tool_use_id"] == "tu_2" and tr[1]["content"] == "7"


# ---------------------------------------------------------- tool error
@pytest.mark.asyncio
async def test_tool_exception_becomes_is_error_result():
    registry = _mini_registry()
    client = _mock_client([
        _turn("tool_use", tool_uses=[
            ToolUse(id="tu_err", name="fail", input={}),
        ]),
        _turn("end_turn", text="apology: the tool failed"),
    ])
    loop = ReActLoop(client, registry, system="s")
    result = await loop.run(task="call fail")

    tr = result.steps[0].tool_results[0]
    assert tr["is_error"] is True
    assert tr["tool_use_id"] == "tu_err"
    assert "RuntimeError" in tr["content"] and "boom" in tr["content"]
    assert result.final_text == "apology: the tool failed"


# ---------------------------------------------------------- message history
@pytest.mark.asyncio
async def test_message_history_contract():
    registry = _mini_registry()
    client = _mock_client([
        _turn("tool_use", tool_uses=[ToolUse(id="tu_1", name="get_time", input={})]),
        _turn("end_turn", text="ok"),
    ])
    loop = ReActLoop(client, registry, system="s")
    result = await loop.run(task="hi")

    # messages: [user: task, assistant: turn1 content, user: tool_results]
    assert len(result.messages) == 3
    assert result.messages[0] == {"role": "user", "content": "hi"}
    assert result.messages[1]["role"] == "assistant"
    assert isinstance(result.messages[1]["content"], list)
    assert result.messages[2]["role"] == "user"
    assert result.messages[2]["content"][0]["type"] == "tool_result"


# ---------------------------------------------------------- no tools at all
@pytest.mark.asyncio
async def test_end_turn_on_first_turn():
    registry = _mini_registry()
    client = _mock_client([_turn("end_turn", text="hello")])
    loop = ReActLoop(client, registry, system="s")
    result = await loop.run(task="say hi")
    assert result.final_text == "hello"
    assert result.step_count == 1
    assert result.steps[0].tool_results == []


# ---------------------------------------------------------- max_tokens early exit
@pytest.mark.asyncio
async def test_max_tokens_returns_partial():
    registry = _mini_registry()
    client = _mock_client([_turn("max_tokens", text="half an answer")])
    loop = ReActLoop(client, registry, system="s")
    result = await loop.run(task="long task")
    assert result.stop_reason == "max_tokens"
    assert result.final_text == "half an answer"


# ---------------------------------------------------------- max_steps
@pytest.mark.asyncio
async def test_max_steps_exceeded():
    registry = _mini_registry()
    # always emit tool_use so the loop never ends naturally
    tool_use_turns = [
        _turn("tool_use", tool_uses=[ToolUse(id=f"t{i}", name="get_time", input={})])
        for i in range(10)
    ]
    client = _mock_client(tool_use_turns)
    loop = ReActLoop(client, registry, system="s", max_steps=3)

    with pytest.raises(MaxStepsExceededError, match="max_steps=3"):
        await loop.run(task="never ends")


# ---------------------------------------------------------- usage aggregation
@pytest.mark.asyncio
async def test_usage_aggregated_across_steps():
    registry = _mini_registry()
    t1 = _turn("tool_use", tool_uses=[ToolUse(id="a", name="get_time", input={})])
    t1.usage = LLMUsage(input=100, output=20, cache_creation=50, cache_read=0)
    t1.cost_usd = 0.001
    t2 = _turn("end_turn", text="ok")
    t2.usage = LLMUsage(input=50, output=10, cache_creation=0, cache_read=50)
    t2.cost_usd = 0.0005

    client = _mock_client([t1, t2])
    loop = ReActLoop(client, registry, system="s")
    result = await loop.run(task="go")

    assert result.usage.input == 150
    assert result.usage.output == 30
    assert result.usage.cache_creation == 50
    assert result.usage.cache_read == 50
    assert result.cost_usd == pytest.approx(0.0015)


# ---------------------------------------------------------- serializer edge cases
@pytest.mark.asyncio
async def test_tool_result_serialization_non_str():
    registry = ToolRegistry()
    registry.register(Tool(
        name="ret_dict",
        description="Return a dict",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=lambda: {"a": 1, "b": [2, 3]},
    ))
    client = _mock_client([
        _turn("tool_use", tool_uses=[ToolUse(id="tu", name="ret_dict", input={})]),
        _turn("end_turn", text="done"),
    ])
    loop = ReActLoop(client, registry, system="s")
    result = await loop.run(task="get dict")

    tr = result.steps[0].tool_results[0]
    assert tr["content"] == '{"a": 1, "b": [2, 3]}'
    assert tr["is_error"] is False
