"""Unit tests for agents_core.tools.common.customer_timeline (task 2.2)."""

from __future__ import annotations

import httpx
import pytest

from agents_core.tools.common.bitrix import BitrixClient
from agents_core.tools.common.customer_timeline import make_customer_timeline_tools

BASE = "https://p.bitrix24.ru/rest/1/tok/"


def _mock(routes: dict[str, dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        segment = request.url.path.split("/")[-1].replace(".json", "")
        if segment not in routes:
            return httpx.Response(404, json={"error": f"no_route:{segment}"})
        return httpx.Response(200, json=routes[segment])

    return httpx.MockTransport(handler)


class TestFactoryShape:
    def test_three_tools(self):
        bx = BitrixClient(BASE)
        names = [t.name for t in make_customer_timeline_tools(bx)]
        assert names == [
            "query_customer_timeline",
            "get_contact_history",
            "search_deals_by_filter",
        ]

    def test_all_read_tier(self):
        bx = BitrixClient(BASE)
        for t in make_customer_timeline_tools(bx):
            assert t.tier == "read"


@pytest.mark.asyncio
class TestQueryTimeline:
    async def test_empty_when_contact_has_no_deals(self):
        client = httpx.AsyncClient(transport=_mock({"crm.deal.list": {"result": []}}))
        bx = BitrixClient(BASE, client=client)
        tool = next(
            t for t in make_customer_timeline_tools(bx) if t.name == "query_customer_timeline"
        )
        result = await tool.handler(contact_id=42)
        assert result == {"contact_id": 42, "deals_count": 0, "events": []}

    async def test_merges_comments_and_activities(self):
        routes = {
            "crm.deal.list": {
                "result": [
                    {
                        "ID": "100",
                        "TITLE": "Deal A",
                        "STAGE_ID": "NEW",
                        "DATE_CREATE": "2026-04-20",
                    },
                ]
            },
            "crm.timeline.comment.list": {
                "result": [
                    {
                        "ID": 1,
                        "CREATED": "2026-04-22T10:00:00",
                        "COMMENT": "whatsapp от клиента",
                        "AUTHOR_ID": 1,
                    }
                ]
            },
            "crm.activity.list": {
                "result": [
                    {
                        "ID": 9,
                        "CREATED": "2026-04-21T15:00:00",
                        "SUBJECT": "Звонок 5 мин",
                        "TYPE_ID": 2,
                        "DESCRIPTION": "clear",
                    }
                ]
            },
        }
        client = httpx.AsyncClient(transport=_mock(routes))
        bx = BitrixClient(BASE, client=client)
        tool = next(
            t for t in make_customer_timeline_tools(bx) if t.name == "query_customer_timeline"
        )
        result = await tool.handler(contact_id=42, limit=5)
        assert result["deals_count"] == 1
        # Two events merged, newest first (2026-04-22 before 2026-04-21)
        assert len(result["events"]) == 2
        assert result["events"][0]["type"] == "timeline_comment"
        assert result["events"][0]["text"] == "whatsapp от клиента"
        assert result["events"][1]["type"] == "activity"
        assert result["events"][1]["subject"] == "Звонок 5 мин"

    async def test_limit_trims_events(self):
        # 1 deal, 5 comments → limit=2 gives 2 events
        routes = {
            "crm.deal.list": {"result": [{"ID": "1", "TITLE": "D", "DATE_CREATE": "2026-04-01"}]},
            "crm.timeline.comment.list": {
                "result": [
                    {"ID": i, "CREATED": f"2026-04-2{i}T10:00:00", "COMMENT": f"c{i}"}
                    for i in range(5)
                ]
            },
            "crm.activity.list": {"result": []},
        }
        client = httpx.AsyncClient(transport=_mock(routes))
        bx = BitrixClient(BASE, client=client)
        tool = next(
            t for t in make_customer_timeline_tools(bx) if t.name == "query_customer_timeline"
        )
        result = await tool.handler(contact_id=42, limit=2)
        assert len(result["events"]) == 2


@pytest.mark.asyncio
class TestContactHistory:
    async def test_summarises_deals_by_stage(self):
        routes = {
            "crm.contact.get": {
                "result": {
                    "NAME": "Иван",
                    "LAST_NAME": "Петров",
                    "PHONE": [{"VALUE": "+7900", "VALUE_TYPE": "WORK"}],
                    "EMAIL": [{"VALUE": "ivan@example.com"}],
                }
            },
            "crm.deal.list": {
                "result": [
                    {
                        "ID": "1",
                        "STAGE_ID": "NEW",
                        "OPPORTUNITY": "10000",
                        "DATE_CREATE": "2026-04-20",
                    },
                    {"ID": "2", "STAGE_ID": "NEW", "DATE_CREATE": "2026-04-18"},
                    {"ID": "3", "STAGE_ID": "WON", "DATE_CREATE": "2026-04-10"},
                ]
            },
        }
        client = httpx.AsyncClient(transport=_mock(routes))
        bx = BitrixClient(BASE, client=client)
        tool = next(
            t for t in make_customer_timeline_tools(bx) if t.name == "get_contact_history"
        )
        result = await tool.handler(contact_id=42)
        assert result["name"].startswith("Иван")
        assert result["phone"] == "+7900"
        assert result["email"] == "ivan@example.com"
        assert result["deals_total"] == 3
        assert result["deals_by_stage"] == {"NEW": 2, "WON": 1}
        assert result["last_touch_at"] == "2026-04-20"


@pytest.mark.asyncio
class TestSearchDeals:
    async def test_passes_filter_through(self):
        captured: dict = {}

        async def h(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content.decode()
            return httpx.Response(
                200, json={"result": [{"ID": "1", "STAGE_ID": "C49:WON"}]}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        bx = BitrixClient(BASE, client=client)
        tool = next(
            t for t in make_customer_timeline_tools(bx) if t.name == "search_deals_by_filter"
        )
        result = await tool.handler(filter={"STAGE_ID": "C49:WON"})
        assert result["matched_count"] == 1
        # Filter was passed to Bitrix as-is
        assert '"STAGE_ID":"C49:WON"' in captured["body"]

    async def test_limit_applied(self):
        routes = {
            "crm.deal.list": {
                "result": [{"ID": str(i), "STAGE_ID": "NEW"} for i in range(100)]
            }
        }
        client = httpx.AsyncClient(transport=_mock(routes))
        bx = BitrixClient(BASE, client=client)
        tool = next(
            t for t in make_customer_timeline_tools(bx) if t.name == "search_deals_by_filter"
        )
        result = await tool.handler(filter={}, limit=5)
        assert result["matched_count"] == 100
        assert len(result["deals"]) == 5
