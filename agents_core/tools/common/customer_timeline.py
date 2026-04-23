"""Customer timeline tools for agents (task 2.2).

Three Bitrix-backed read tools the briefing agent and client-bot need to
answer "what has this customer been through":

- ``query_customer_timeline(contact_id, limit)``  combined timeline
  comments + activities (WhatsApp, calls, meetings, tasks).
- ``get_contact_history(contact_id)``  compact history: deals summary,
  last contact channel, last message.
- ``search_deals_by_filter(filter)``  flexible deal search (stage,
  assigned_by, date range, UF fields) — batched across pages.

All tools are ``tier="read"`` and idempotent. They extend the Bitrix
toolset from ``agents_core.tools.common.bitrix``; callers pass the same
``BitrixClient`` instance so connections / auth are shared.

Shape contract (per research §9): tool_result is always a JSON object
with a stable set of keys. Empty timeline → ``{"events": []}``, not
``None`` — the model must not need to disambiguate null vs empty.
"""

from __future__ import annotations

from typing import Any

from agents_core.tools.common.bitrix import BitrixClient
from agents_core.tools.registry import Tool

__all__ = ["make_customer_timeline_tools"]


async def _query_timeline(
    bx: BitrixClient, contact_id: int, limit: int
) -> dict[str, Any]:
    """Fetch timeline comments + activities for all deals of this contact.

    Bitrix API: crm.deal.list by CONTACT_ID → per-deal crm.timeline.comment.list
    and crm.activity.list. We keep the loop tight (at most 20 deals) to avoid
    blowing the tool_result size past ~4K chars that the agent can reason over.
    """
    deals = await bx.call(
        "crm.deal.list",
        {
            "filter": {"CONTACT_ID": int(contact_id)},
            "select": ["ID", "TITLE", "STAGE_ID", "DATE_CREATE", "ASSIGNED_BY_ID"],
            "order": {"DATE_CREATE": "DESC"},
        },
    )
    if not isinstance(deals, list):
        deals = []
    deals = deals[:20]

    events: list[dict[str, Any]] = []
    for d in deals:
        deal_id = d.get("ID")
        # Timeline comments (WhatsApp, manual notes, Wazzup)
        comments = await bx.call(
            "crm.timeline.comment.list",
            {
                "filter": {"ENTITY_ID": int(deal_id), "ENTITY_TYPE": "deal"},
                "order": {"CREATED": "DESC"},
                "select": ["ID", "CREATED", "COMMENT", "AUTHOR_ID"],
            },
        )
        for c in (comments or [])[:5]:
            events.append(
                {
                    "type": "timeline_comment",
                    "deal_id": deal_id,
                    "deal_title": d.get("TITLE"),
                    "at": c.get("CREATED"),
                    "author_id": c.get("AUTHOR_ID"),
                    "text": (c.get("COMMENT") or "")[:500],
                }
            )
        # Activities (calls, meetings, WhatsApp sent via OpenLines)
        acts = await bx.call(
            "crm.activity.list",
            {
                "filter": {"OWNER_ID": int(deal_id), "OWNER_TYPE_ID": "2"},  # 2=Deal
                "order": {"CREATED": "DESC"},
                "select": ["ID", "CREATED", "SUBJECT", "TYPE_ID", "DESCRIPTION"],
            },
        )
        for a in (acts or [])[:5]:
            events.append(
                {
                    "type": "activity",
                    "activity_type": a.get("TYPE_ID"),
                    "deal_id": deal_id,
                    "deal_title": d.get("TITLE"),
                    "at": a.get("CREATED"),
                    "subject": (a.get("SUBJECT") or "")[:200],
                }
            )

    events.sort(key=lambda e: e.get("at") or "", reverse=True)
    return {
        "contact_id": int(contact_id),
        "deals_count": len(deals),
        "events": events[:limit],
    }


async def _get_contact_history(
    bx: BitrixClient, contact_id: int
) -> dict[str, Any]:
    """Compact contact summary: deals by stage + last known touch."""
    contact = await bx.call("crm.contact.get", {"ID": int(contact_id)})
    deals = await bx.call(
        "crm.deal.list",
        {
            "filter": {"CONTACT_ID": int(contact_id)},
            "select": ["ID", "TITLE", "STAGE_ID", "OPPORTUNITY", "DATE_CREATE"],
            "order": {"DATE_CREATE": "DESC"},
        },
    )
    deals = deals if isinstance(deals, list) else []
    by_stage: dict[str, int] = {}
    for d in deals:
        s = str(d.get("STAGE_ID") or "UNKNOWN")
        by_stage[s] = by_stage.get(s, 0) + 1
    last_touch_at = max(
        (d.get("DATE_CREATE") for d in deals if d.get("DATE_CREATE")),
        default=None,
    )
    return {
        "contact_id": int(contact_id),
        "name": (contact.get("NAME") or "") + " " + (contact.get("LAST_NAME") or ""),
        "phone": _first_multi_field(contact, "PHONE"),
        "email": _first_multi_field(contact, "EMAIL"),
        "deals_total": len(deals),
        "deals_by_stage": by_stage,
        "last_touch_at": last_touch_at,
    }


def _first_multi_field(contact: dict, key: str) -> str | None:
    """Bitrix contact phones/emails arrive as lists of {VALUE, TYPE} dicts."""
    raw = contact.get(key)
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict):
            return first.get("VALUE")
    if isinstance(raw, str):
        return raw
    return None


async def _search_deals(
    bx: BitrixClient,
    filter_: dict[str, Any] | None,
    select: list[str] | None,
    limit: int,
) -> dict[str, Any]:
    """Flexible deal search. `filter_` is passed directly to Bitrix.

    The agent can pass stage/date/UF filters in one shot — Bitrix CRM
    API validates unknown fields itself.
    """
    deals = await bx.call(
        "crm.deal.list",
        {
            "filter": filter_ or {},
            "select": select or ["ID", "TITLE", "STAGE_ID", "DATE_CREATE", "OPPORTUNITY"],
            "order": {"DATE_CREATE": "DESC"},
        },
    )
    deals = deals if isinstance(deals, list) else []
    return {
        "matched_count": len(deals),
        "deals": deals[:limit],
    }


def make_customer_timeline_tools(bx: BitrixClient) -> list[Tool]:
    return [
        Tool(
            name="query_customer_timeline",
            description=(
                "Return a merged timeline (comments + activities) across all "
                "deals of a contact, newest first. Use when asked about a "
                "customer's recent interactions or last touch."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "contact_id": {"type": "integer"},
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["contact_id"],
            },
            handler=lambda contact_id, limit=20, _bx=bx: _query_timeline(
                _bx, contact_id, limit
            ),
            tier="read",
            tags=("bitrix", "timeline", "read"),
        ),
        Tool(
            name="get_contact_history",
            description=(
                "Compact contact summary: name, phone, email, total deals, "
                "deals grouped by stage_id, last touch date. Use when "
                "briefly situating a contact before replying."
            ),
            input_schema={
                "type": "object",
                "properties": {"contact_id": {"type": "integer"}},
                "required": ["contact_id"],
            },
            handler=lambda contact_id, _bx=bx: _get_contact_history(_bx, contact_id),
            tier="read",
            tags=("bitrix", "contact", "read"),
        ),
        Tool(
            name="search_deals_by_filter",
            description=(
                "Find deals by arbitrary Bitrix filter (e.g. {'STAGE_ID': "
                "'C49:WON'} or {'ASSIGNED_BY_ID': 15}). Returns matched_count "
                "+ up to `limit` deals. Use when the briefing asks for deals "
                "by owner, stage, or date range."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "object",
                        "description": (
                            "Bitrix filter dict. Common keys: STAGE_ID, "
                            "ASSIGNED_BY_ID, CATEGORY_ID, >=DATE_CREATE, "
                            "<=DATE_CREATE, CONTACT_ID."
                        ),
                    },
                    "select": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Fields to return (default: ID, TITLE, "
                            "STAGE_ID, DATE_CREATE, OPPORTUNITY)."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 200,
                    },
                },
                "required": ["filter"],
            },
            handler=lambda filter, select=None, limit=50, _bx=bx: _search_deals(
                _bx, filter, select, limit
            ),
            tier="read",
            tags=("bitrix", "deals", "read"),
        ),
    ]
