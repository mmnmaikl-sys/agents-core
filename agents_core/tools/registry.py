"""Tool registry — declarative catalog of Anthropic-compatible tools.

Core primitives (research/architecture.md §4 Tool Registry, §11 principles 2 and 5):
- `Tool` — frozen dataclass bundling name / description / JSON schema / handler /
  tier / idempotency / verify-pair requirement. Tier drives safety behaviour:
      read    — side-effect-free, always callable
      write   — state-changing, should have a paired `*_verify` tool and an
                idempotency_key per research §9 rules 1 and 4
      danger  — destructive (delete/close_won), requires `user_confirmed=true`
                before execution per research §9 rule 6.
- `ToolRegistry` — owns a set of Tool instances, emits Anthropic-API-compatible
  `for_api()` payload, supports filtering by tier / tags so each agent can hand
  Claude only the subset it is trusted to call.

`for_api()` returns a list of dicts matching the Anthropic messages API schema:
    {"name": ..., "description": ..., "input_schema": {...}}
Feed straight into `client.messages.create(tools=registry.for_api(...))`.

The runner is intentionally tiny (`call(name, **kwargs)`): the real ReAct loop
lands in `loop/react.py` (task 0.7). Here we only cover (a) metadata, (b) API
projection, (c) safe handler dispatch.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

Tier = Literal["read", "write", "danger"]

_VALID_TIERS: frozenset[str] = frozenset(("read", "write", "danger"))


@dataclass(frozen=True)
class Tool:
    """Declarative tool descriptor.

    input_schema must be a valid JSON schema object (Anthropic format). Must
    have at least `type: object` so that Claude can infer the argument shape.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    tier: Tier = "read"
    idempotent: bool = True
    requires_verify: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.tier not in _VALID_TIERS:
            raise ValueError(f"tier must be one of {sorted(_VALID_TIERS)}, got {self.tier!r}")
        if self.input_schema.get("type") != "object":
            raise ValueError(
                f"{self.name}: input_schema.type must be 'object' for Anthropic API compatibility"
            )
        if self.tier == "danger" and self.requires_verify is False:
            # danger tools without verify are allowed but noisy — no hard error
            pass

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize to Anthropic messages API tool shape."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolAlreadyRegisteredError(ValueError):
    pass


class ToolNotFoundError(KeyError):
    pass


class ToolRegistry:
    """Keyed collection of Tool instances with filter + API projection."""

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        if tools:
            for t in tools:
                self.register(t)

    # --- registry ops ------------------------------------------------
    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ToolAlreadyRegisteredError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"no tool named {name!r}") from exc

    def __getitem__(self, name: str) -> Tool:
        return self.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    # --- filtering ---------------------------------------------------
    def filter(
        self,
        *,
        tiers: Iterable[Tier] | None = None,
        tags: Iterable[str] | None = None,
        names: Iterable[str] | None = None,
    ) -> list[Tool]:
        """Return Tools matching ALL provided criteria. None → no filter on axis."""
        tier_set = set(tiers) if tiers is not None else None
        tag_set = set(tags) if tags is not None else None
        name_set = set(names) if names is not None else None

        out: list[Tool] = []
        for t in self._tools.values():
            if tier_set is not None and t.tier not in tier_set:
                continue
            if tag_set is not None and not tag_set.intersection(t.tags):
                continue
            if name_set is not None and t.name not in name_set:
                continue
            out.append(t)
        return out

    # --- API projection ---------------------------------------------
    def for_api(
        self,
        *,
        tiers: Iterable[Tier] | None = None,
        tags: Iterable[str] | None = None,
        names: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Project filtered tools to Anthropic messages API format."""
        return [t.to_api_dict() for t in self.filter(tiers=tiers, tags=tags, names=names)]

    # --- dispatch ---------------------------------------------------
    async def call(self, name: str, **kwargs: Any) -> Any:
        """Invoke the handler by name. Awaits coroutines, returns sync results as-is.

        Tier-specific guardrails (HITL for danger, verify-pair for write) land
        in loop/react.py — this method is the bare-metal dispatch used by tests
        and by the loop once policy checks pass.
        """
        tool = self.get(name)
        result = tool.handler(**kwargs)
        if inspect.iscoroutine(result):
            result = await result
        return result
