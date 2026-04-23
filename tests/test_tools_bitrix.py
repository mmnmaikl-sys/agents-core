"""Unit tests for agents_core.tools.common.bitrix (task 0.18)."""

from __future__ import annotations

import httpx
import pytest

from agents_core.tools.common.bitrix import (
    BitrixClient,
    BitrixError,
    make_bitrix_tools,
)

BASE = "https://p.bitrix24.ru/rest/1/tok/"


def _mock_transport(routes: dict[str, dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.split("/")[-1].replace(".json", "")
        if path not in routes:
            return httpx.Response(404, json={"error": "no_route", "error_description": path})
        return httpx.Response(200, json=routes[path])
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
class TestBitrixClient:
    async def test_calls_method_and_returns_result(self):
        client = httpx.AsyncClient(
            transport=_mock_transport({"crm.deal.get": {"result": {"ID": 1}}})
        )
        bx = BitrixClient(BASE, client=client)
        assert await bx.call("crm.deal.get", {"ID": 1}) == {"ID": 1}

    async def test_normalises_trailing_slash(self):
        # no trailing slash → still works
        bx = BitrixClient(BASE.rstrip("/"))
        assert bx._base.endswith("/")

    async def test_raises_bitrix_error(self):
        client = httpx.AsyncClient(
            transport=_mock_transport(
                {"crm.deal.get": {"error": "ACCESS_DENIED", "error_description": "no"}}
            )
        )
        bx = BitrixClient(BASE, client=client)
        with pytest.raises(BitrixError) as ei:
            await bx.call("crm.deal.get", {"ID": 1})
        assert ei.value.code == "ACCESS_DENIED"


@pytest.mark.asyncio
class TestFactoryShape:
    async def test_eleven_tools_including_user_search(self):
        bx = BitrixClient(BASE)
        tools = make_bitrix_tools(bx)
        # 8 base + 1 verify pair + deal_close_won + user_search (wave 2)
        names = [t.name for t in tools]
        for required in (
            "bitrix_deal_get",
            "bitrix_deal_update",
            "bitrix_deal_update_verify",
            "bitrix_deal_close_won",
            "bitrix_user_search",
        ):
            assert required in names, required
        assert len(tools) == 11

    async def test_close_won_is_danger_tier(self):
        bx = BitrixClient(BASE)
        tools = {t.name: t for t in make_bitrix_tools(bx)}
        assert tools["bitrix_deal_close_won"].tier == "danger"
        assert "user_confirmed" in tools["bitrix_deal_close_won"].input_schema["required"]


@pytest.mark.asyncio
class TestDealUpdateAndVerify:
    async def test_update_stashes_idempotency_key_in_comment(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"result": True})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        bx = BitrixClient(BASE, client=client)
        update = next(
            t for t in make_bitrix_tools(bx) if t.name == "bitrix_deal_update"
        )
        await update.handler(
            deal_id=10,
            fields={"TITLE": "X"},
            idempotency_key="abc-123",
        )
        assert "abc-123" in captured["body"]
        assert '"TITLE":"X"' in captured["body"]

    async def test_verify_reports_diff(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"result": {"TITLE": "X", "STAGE_ID": "NEW", "ID": 10}},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        bx = BitrixClient(BASE, client=client)
        verify = next(
            t for t in make_bitrix_tools(bx)
            if t.name == "bitrix_deal_update_verify"
        )
        result = await verify.handler(
            deal_id=10,
            expected={"TITLE": "X", "STAGE_ID": "WON"},
        )
        assert result["verified"] is False
        assert result["diff"] == {"STAGE_ID": ("NEW", "WON")}

    async def test_verify_reports_all_ok(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"result": {"TITLE": "X"}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        bx = BitrixClient(BASE, client=client)
        verify = next(
            t for t in make_bitrix_tools(bx)
            if t.name == "bitrix_deal_update_verify"
        )
        result = await verify.handler(deal_id=10, expected={"TITLE": "X"})
        assert result == {"verified": True, "diff": {}}


@pytest.mark.asyncio
class TestReadTools:
    async def test_contact_get(self):
        client = httpx.AsyncClient(
            transport=_mock_transport({"crm.contact.get": {"result": {"ID": 1}}})
        )
        bx = BitrixClient(BASE, client=client)
        get = next(t for t in make_bitrix_tools(bx) if t.name == "bitrix_contact_get")
        assert await get.handler(contact_id=1) == {"ID": 1}

    async def test_user_get(self):
        client = httpx.AsyncClient(
            transport=_mock_transport({"user.get": {"result": [{"ID": "1"}]}})
        )
        bx = BitrixClient(BASE, client=client)
        get = next(t for t in make_bitrix_tools(bx) if t.name == "bitrix_user_get")
        assert await get.handler(user_id=1) == [{"ID": "1"}]
