"""Unit tests for agents_core.tools.registry.

Covers:
- Tool dataclass validation (tier whitelist, input_schema.type==object)
- ToolRegistry register / get / __contains__ / __iter__ / __len__
- Duplicate registration raises
- Filter by tier, tags, names; intersections and empty results
- for_api() returns valid Anthropic schema list (the DoD 5-tool case)
- call() dispatch for sync and async handlers
"""
from __future__ import annotations

import pytest

from agents_core.tools import (
    Tool,
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolRegistry,
)


# ---------------------------------------------------------- fixtures
def _schema(*required: str, **props: str) -> dict:
    if not props:
        props = {r: "string" for r in required}
    return {
        "type": "object",
        "properties": {k: {"type": v} for k, v in props.items()},
        "required": list(required),
    }


def _tool(name: str, tier: str = "read", tags: tuple = (), **extra) -> Tool:
    return Tool(
        name=name,
        description=f"description of {name}",
        input_schema=_schema("arg"),
        handler=lambda **kw: {"called": name, **kw},
        tier=tier,
        tags=tags,
        **extra,
    )


@pytest.fixture
def five_tools() -> list[Tool]:
    """DoD fixture: 5 tools across tiers and tags."""
    return [
        Tool(
            name="deal_get",
            description="Fetch a Bitrix deal by id.",
            input_schema=_schema("deal_id"),
            handler=lambda deal_id: {"id": deal_id, "stage": "C49:NEW"},
            tier="read",
            tags=("bitrix",),
        ),
        Tool(
            name="deal_update",
            description="Update deal fields. Must be followed by deal_get_verify.",
            input_schema={
                "type": "object",
                "properties": {
                    "deal_id": {"type": "string"},
                    "fields": {"type": "object"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["deal_id", "fields", "idempotency_key"],
            },
            handler=lambda deal_id, fields, idempotency_key: {"ok": True, "deal_id": deal_id},
            tier="write",
            idempotent=True,
            requires_verify=True,
            tags=("bitrix",),
        ),
        Tool(
            name="deal_close_lost",
            description="Set deal to lost stage. Destructive.",
            input_schema=_schema("deal_id", "reason"),
            handler=lambda deal_id, reason: {"closed": deal_id},
            tier="danger",
            requires_verify=True,
            tags=("bitrix",),
        ),
        Tool(
            name="rag_search",
            description="Semantic search over project knowledge.",
            input_schema=_schema("query"),
            handler=lambda query: [{"chunk": "match for " + query}],
            tier="read",
            tags=("rag",),
        ),
        Tool(
            name="tg_notify_admin",
            description="Send a Telegram message to admin chat.",
            input_schema=_schema("text"),
            handler=lambda text: {"sent": True, "text": text},
            tier="write",
            requires_verify=False,
            tags=("telegram",),
        ),
    ]


@pytest.fixture
def registry(five_tools: list[Tool]) -> ToolRegistry:
    return ToolRegistry(five_tools)


# ---------------------------------------------------------- Tool validation
def test_tool_invalid_tier_raises():
    with pytest.raises(ValueError, match="tier must be one of"):
        _tool("x", tier="evil")


def test_tool_invalid_schema_type_raises():
    with pytest.raises(ValueError, match="input_schema.type must be 'object'"):
        Tool(
            name="broken",
            description="",
            input_schema={"type": "array"},
            handler=lambda: None,
        )


def test_tool_frozen():
    import dataclasses

    t = _tool("a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.name = "b"  # type: ignore[misc]


def test_tool_to_api_dict_shape():
    t = _tool("x")
    d = t.to_api_dict()
    assert set(d.keys()) == {"name", "description", "input_schema"}
    assert d["input_schema"]["type"] == "object"
    # tier/handler/tags must NOT leak to Anthropic payload
    assert "tier" not in d and "handler" not in d


# ---------------------------------------------------------- registry basics
def test_registry_len_iter_contains(registry: ToolRegistry):
    assert len(registry) == 5
    names = [t.name for t in registry]
    assert set(names) == {
        "deal_get", "deal_update", "deal_close_lost", "rag_search", "tg_notify_admin",
    }
    assert "deal_get" in registry
    assert "nonexistent" not in registry


def test_registry_get_and_getitem(registry: ToolRegistry):
    assert registry.get("deal_get").name == "deal_get"
    assert registry["rag_search"].name == "rag_search"


def test_registry_duplicate_raises():
    r = ToolRegistry()
    r.register(_tool("x"))
    with pytest.raises(ToolAlreadyRegisteredError):
        r.register(_tool("x"))


def test_registry_get_missing_raises(registry: ToolRegistry):
    with pytest.raises(ToolNotFoundError):
        registry.get("nope")


def test_empty_registry_len_zero():
    assert len(ToolRegistry()) == 0


# ---------------------------------------------------------- filtering
def test_filter_by_tier_read(registry: ToolRegistry):
    reads = [t.name for t in registry.filter(tiers=["read"])]
    assert set(reads) == {"deal_get", "rag_search"}


def test_filter_by_tier_write_and_danger(registry: ToolRegistry):
    hot = [t.name for t in registry.filter(tiers=["write", "danger"])]
    assert set(hot) == {"deal_update", "deal_close_lost", "tg_notify_admin"}


def test_filter_by_tag(registry: ToolRegistry):
    bitrix = [t.name for t in registry.filter(tags=["bitrix"])]
    assert set(bitrix) == {"deal_get", "deal_update", "deal_close_lost"}


def test_filter_by_names(registry: ToolRegistry):
    subset = [t.name for t in registry.filter(names=["rag_search", "tg_notify_admin"])]
    assert set(subset) == {"rag_search", "tg_notify_admin"}


def test_filter_intersection(registry: ToolRegistry):
    # bitrix AND read → only deal_get
    result = [t.name for t in registry.filter(tiers=["read"], tags=["bitrix"])]
    assert result == ["deal_get"]


def test_filter_empty_result(registry: ToolRegistry):
    # telegram AND danger → none
    assert registry.filter(tiers=["danger"], tags=["telegram"]) == []


# ---------------------------------------------------------- for_api (DoD)
def test_for_api_returns_anthropic_schema_5_tools(registry: ToolRegistry):
    """DoD: 5 tools, registry.for_api() = valid Anthropic tools JSON."""
    api = registry.for_api()
    assert len(api) == 5
    for tool_dict in api:
        assert set(tool_dict.keys()) == {"name", "description", "input_schema"}
        assert isinstance(tool_dict["name"], str) and tool_dict["name"]
        assert isinstance(tool_dict["description"], str) and tool_dict["description"]
        assert tool_dict["input_schema"]["type"] == "object"
        assert "properties" in tool_dict["input_schema"]

    # Verify the write tool exposes idempotency_key in required (per research §9 rule 4)
    upd = next(x for x in api if x["name"] == "deal_update")
    assert "idempotency_key" in upd["input_schema"]["required"]


def test_for_api_with_filter(registry: ToolRegistry):
    read_only = registry.for_api(tiers=["read"])
    assert {d["name"] for d in read_only} == {"deal_get", "rag_search"}


def test_for_api_empty_registry():
    assert ToolRegistry().for_api() == []


# ---------------------------------------------------------- dispatch
@pytest.mark.asyncio
async def test_call_sync_handler(registry: ToolRegistry):
    result = await registry.call("deal_get", deal_id="123")
    assert result == {"id": "123", "stage": "C49:NEW"}


@pytest.mark.asyncio
async def test_call_async_handler():
    async def handler(x: int) -> int:
        return x * 2

    r = ToolRegistry()
    r.register(Tool(
        name="double",
        description="Double a number",
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
        handler=handler,
    ))
    assert await r.call("double", x=21) == 42


@pytest.mark.asyncio
async def test_call_missing_tool(registry: ToolRegistry):
    with pytest.raises(ToolNotFoundError):
        await registry.call("nope", x=1)
