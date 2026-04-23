"""HITL (human-in-the-loop) gate for destructive tools (research §9 rule 6).

``danger``-tier tools must not execute until the caller presents
``user_confirmed=True`` in the input. The gate is a pure check that lives
between the ReAct loop and ``ToolRegistry.call``: orchestrators that want
HITL wrap tool calls in ``require_confirmation`` and surface a "пришли
подтверждение" message to the operator.

Design notes
------------
- This is *authorisation*, not authentication. The caller is trusted to
  only set ``user_confirmed=True`` after getting a real human OK; the
  agent shouldn't flip it on its own. Downstream audit (observability/
  audit.py) logs who confirmed, so post-hoc review is possible.
- Read tools never need confirmation. Write tools get it only via an
  explicit ``require_for_write=True``; default is off to avoid friction
  in batch automation (sync jobs, bulk updates) that have their own
  compensating controls.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agents_core.tools.registry import Tool

__all__ = ["ConfirmationRequired", "require_confirmation"]


class ConfirmationRequired(PermissionError):  # noqa: N818  # domain term, not "Error"
    """Raised when a tool requires a confirmation flag that's missing."""

    def __init__(self, tool_name: str, tier: str) -> None:
        super().__init__(
            f"{tool_name!r} ({tier}) requires user_confirmed=true before execution"
        )
        self.tool_name = tool_name
        self.tier = tier


def require_confirmation(
    tool: Tool,
    inputs: Mapping[str, Any],
    *,
    require_for_write: bool = False,
    flag_name: str = "user_confirmed",
) -> None:
    """Raise ``ConfirmationRequired`` if ``tool`` needs HITL and the flag is missing.

    ``inputs`` is what's about to be passed to the tool handler. We only
    read ``inputs[flag_name]`` — never mutate or strip it; handlers often
    want to see the same flag for their own audit.
    """
    if tool.tier == "read":
        return
    if tool.tier == "write" and not require_for_write:
        return
    confirmed = bool(inputs.get(flag_name, False))
    if not confirmed:
        raise ConfirmationRequired(tool.name, tool.tier)
