"""Yandex.Direct execution tools (task 1.2 — CMO agent).

Six tools for Direct API v5:
- ``direct_campaigns_get``        read: list campaigns + state
- ``direct_campaign_pause``       write+verify: suspend one or more campaigns
- ``direct_campaign_resume``      write+verify: resume campaigns
- ``direct_campaign_update_budget`` write+verify: change DailyBudget.Amount
- ``direct_keywords_get``         read: list keywords in an ad group
- ``direct_keyword_update_bid``   write+verify: change keyword Bid

All write-tools are ``tier="write"`` with ``requires_verify=True`` — the
paired VERIFY tool in this module (``direct_campaign_verify_state``,
``direct_keyword_verify_bid``) re-reads the entity and confirms expected
value. Per research §9 rule 1, the loop must call verify before reporting
success.

The client takes only the raw OAuth token. Auth header is ``Bearer``;
Direct accepts both ``Bearer`` and ``OAuth`` (vs Metrika which requires
``OAuth`` specifically — see ai_cmo_orchestrator_deployed memory gotcha).

``idempotency_key`` is stashed into a per-request note field where the
API allows; for campaigns/keywords there is no such field, so the key
is only used by the caller's audit log, not echoed back to Direct.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from agents_core.tools.registry import Tool

__all__ = ["DirectClient", "make_direct_tools"]

logger = logging.getLogger(__name__)

DIRECT_API_URL = "https://api.direct.yandex.com/json/v5"
_TIMEOUT = 20.0


class DirectError(RuntimeError):
    def __init__(self, code: int | None, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class DirectClient:
    """Thin async wrapper over Direct API v5 JSON endpoints."""

    def __init__(
        self,
        access_token: str,
        client: httpx.AsyncClient | None = None,
        timeout: float = _TIMEOUT,
    ) -> None:
        if not access_token:
            raise ValueError("access_token required")
        self._headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept-Language": "ru",
            "Content-Type": "application/json; charset=utf-8",
        }
        self._client = client
        self._timeout = timeout

    async def call(
        self,
        resource: str,  # e.g. "campaigns" -> POST /json/v5/campaigns
        method: str,  # "get" | "suspend" | "resume" | "update"
        params: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{DIRECT_API_URL}/{resource}"
        body = {"method": method, "params": params}
        c = self._client
        owns_client = c is None
        if owns_client:
            c = httpx.AsyncClient()
        try:
            resp = await c.post(
                url, json=body, headers=self._headers, timeout=self._timeout
            )
            resp.raise_for_status()
            data = resp.json()
        finally:
            if owns_client:
                await c.aclose()
        if "error" in data:
            err = data["error"]
            raise DirectError(err.get("error_code"), err.get("error_string", "unknown"))
        return data.get("result") or {}


# --- handler helpers ---------------------------------------------------

async def _campaigns_get(dc: DirectClient, campaign_ids: list[int] | None):
    criteria: dict[str, Any] = {
        "States": ["ON", "OFF", "SUSPENDED", "ENDED"],
    }
    if campaign_ids:
        criteria["Ids"] = [int(i) for i in campaign_ids]
    return await dc.call(
        "campaigns",
        "get",
        {
            "SelectionCriteria": criteria,
            "FieldNames": ["Id", "Name", "State", "DailyBudget"],
        },
    )


async def _campaign_pause(dc: DirectClient, campaign_ids: list[int]):
    return await dc.call(
        "campaigns", "suspend", {"SelectionCriteria": {"Ids": list(campaign_ids)}}
    )


async def _campaign_resume(dc: DirectClient, campaign_ids: list[int]):
    return await dc.call(
        "campaigns", "resume", {"SelectionCriteria": {"Ids": list(campaign_ids)}}
    )


async def _campaign_update_budget(
    dc: DirectClient, campaign_id: int, daily_budget_rub: int, mode: str
):
    if mode not in ("STANDARD", "DISTRIBUTED"):
        raise ValueError("mode must be STANDARD or DISTRIBUTED")
    return await dc.call(
        "campaigns",
        "update",
        {
            "Campaigns": [
                {
                    "Id": int(campaign_id),
                    "DailyBudget": {
                        "Amount": int(daily_budget_rub) * 1_000_000,  # micros
                        "Mode": mode,
                    },
                }
            ]
        },
    )


async def _campaign_verify_state(
    dc: DirectClient, campaign_id: int, expected_state: str
):
    data = await dc.call(
        "campaigns",
        "get",
        {
            "SelectionCriteria": {"Ids": [int(campaign_id)]},
            "FieldNames": ["Id", "State"],
        },
    )
    cs = data.get("Campaigns") or []
    actual = cs[0].get("State") if cs else None
    return {
        "verified": actual == expected_state,
        "actual_state": actual,
        "expected_state": expected_state,
    }


async def _keywords_get(dc: DirectClient, ad_group_id: int):
    return await dc.call(
        "keywords",
        "get",
        {
            "SelectionCriteria": {"AdGroupIds": [int(ad_group_id)]},
            "FieldNames": ["Id", "Keyword", "State", "Status"],
        },
    )


async def _keyword_update_bid(dc: DirectClient, keyword_id: int, bid_rub: float):
    """Update keyword Bid. Bid is in micros (rub * 1_000_000)."""
    return await dc.call(
        "keywordbids",
        "set",
        {
            "KeywordBids": [
                {
                    "KeywordId": int(keyword_id),
                    "SearchBid": int(round(bid_rub * 1_000_000)),
                }
            ]
        },
    )


async def _keyword_verify_bid(
    dc: DirectClient, keyword_id: int, expected_bid_rub: float, tolerance_rub: float
):
    data = await dc.call(
        "keywordbids",
        "get",
        {
            "SelectionCriteria": {"KeywordIds": [int(keyword_id)]},
            "FieldNames": ["KeywordId", "SearchBid"],
        },
    )
    kb = (data.get("KeywordBids") or [{}])[0]
    actual_micros = kb.get("SearchBid") or 0
    actual_rub = actual_micros / 1_000_000
    return {
        "verified": abs(actual_rub - expected_bid_rub) <= tolerance_rub,
        "actual_bid_rub": round(actual_rub, 2),
        "expected_bid_rub": expected_bid_rub,
    }


# --- factory -----------------------------------------------------------


def make_direct_tools(dc: DirectClient) -> list[Tool]:
    """Return the 6 base Direct tools + 2 verify pairs (8 total).

    Research §9 rule 1: every write gets a paired verify. Rule 4: write
    handlers take ``idempotency_key`` so the caller's audit log can detect
    duplicate invocations; Direct has no idempotency header so the key
    is not echoed to the API.
    """
    return [
        Tool(
            name="direct_campaigns_get",
            description=(
                "List Yandex Direct campaigns. Returns {Campaigns: [{Id, Name, "
                "State, DailyBudget}]}. Pass campaign_ids to filter."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "campaign_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional campaign IDs filter.",
                    }
                },
            },
            handler=lambda campaign_ids=None, _dc=dc: _campaigns_get(_dc, campaign_ids),
            tier="read",
            tags=("direct", "read"),
        ),
        Tool(
            name="direct_campaign_pause",
            description=(
                "Suspend (pause) one or more campaigns. Traffic stops within "
                "~5 minutes. Always call direct_campaign_verify_state after."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "campaign_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                    },
                    "idempotency_key": {"type": "string"},
                },
                "required": ["campaign_ids", "idempotency_key"],
            },
            handler=lambda campaign_ids, idempotency_key, _dc=dc: _campaign_pause(
                _dc, campaign_ids
            ),
            tier="write",
            idempotent=False,
            requires_verify=True,
            tags=("direct", "write"),
        ),
        Tool(
            name="direct_campaign_resume",
            description=(
                "Resume previously suspended campaigns. Always call "
                "direct_campaign_verify_state after."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "campaign_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                    },
                    "idempotency_key": {"type": "string"},
                },
                "required": ["campaign_ids", "idempotency_key"],
            },
            handler=lambda campaign_ids, idempotency_key, _dc=dc: _campaign_resume(
                _dc, campaign_ids
            ),
            tier="write",
            idempotent=False,
            requires_verify=True,
            tags=("direct", "write"),
        ),
        Tool(
            name="direct_campaign_update_budget",
            description=(
                "Change daily budget of a single campaign (in rubles). mode "
                "= STANDARD (fixed) or DISTRIBUTED (adaptive). Always verify."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "campaign_id": {"type": "integer"},
                    "daily_budget_rub": {"type": "integer", "minimum": 100},
                    "mode": {"type": "string", "enum": ["STANDARD", "DISTRIBUTED"]},
                    "idempotency_key": {"type": "string"},
                },
                "required": [
                    "campaign_id",
                    "daily_budget_rub",
                    "mode",
                    "idempotency_key",
                ],
            },
            handler=lambda campaign_id, daily_budget_rub, mode, idempotency_key, _dc=dc: (
                _campaign_update_budget(_dc, campaign_id, daily_budget_rub, mode)
            ),
            tier="write",
            idempotent=False,
            requires_verify=True,
            tags=("direct", "write"),
        ),
        Tool(
            name="direct_campaign_verify_state",
            description=(
                "VERIFY counterpart for pause/resume. Re-reads campaign and "
                "returns {verified: bool, actual_state, expected_state}."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "campaign_id": {"type": "integer"},
                    "expected_state": {
                        "type": "string",
                        "enum": ["ON", "OFF", "SUSPENDED", "ENDED", "ARCHIVED"],
                    },
                },
                "required": ["campaign_id", "expected_state"],
            },
            handler=lambda campaign_id, expected_state, _dc=dc: _campaign_verify_state(
                _dc, campaign_id, expected_state
            ),
            tier="read",
            tags=("direct", "verify"),
        ),
        Tool(
            name="direct_keywords_get",
            description=(
                "List keywords in an ad group. Returns {Keywords: [{Id, "
                "Keyword, State, Status}]}."
            ),
            input_schema={
                "type": "object",
                "properties": {"ad_group_id": {"type": "integer"}},
                "required": ["ad_group_id"],
            },
            handler=lambda ad_group_id, _dc=dc: _keywords_get(_dc, ad_group_id),
            tier="read",
            tags=("direct", "read"),
        ),
        Tool(
            name="direct_keyword_update_bid",
            description=(
                "Change keyword SearchBid (rub). Always call "
                "direct_keyword_verify_bid after."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "keyword_id": {"type": "integer"},
                    "bid_rub": {"type": "number", "minimum": 1},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["keyword_id", "bid_rub", "idempotency_key"],
            },
            handler=lambda keyword_id, bid_rub, idempotency_key, _dc=dc: (
                _keyword_update_bid(_dc, keyword_id, bid_rub)
            ),
            tier="write",
            idempotent=False,
            requires_verify=True,
            tags=("direct", "write"),
        ),
        Tool(
            name="direct_keyword_verify_bid",
            description=(
                "VERIFY counterpart for direct_keyword_update_bid. Re-reads "
                "keyword bid, compares within tolerance_rub (default 0.01)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "keyword_id": {"type": "integer"},
                    "expected_bid_rub": {"type": "number"},
                    "tolerance_rub": {"type": "number", "default": 0.01},
                },
                "required": ["keyword_id", "expected_bid_rub"],
            },
            handler=lambda keyword_id, expected_bid_rub, tolerance_rub=0.01, _dc=dc: (
                _keyword_verify_bid(_dc, keyword_id, expected_bid_rub, tolerance_rub)
            ),
            tier="read",
            tags=("direct", "verify"),
        ),
    ]
