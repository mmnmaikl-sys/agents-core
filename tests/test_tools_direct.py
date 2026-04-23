"""Unit tests for agents_core.tools.common.direct (task 1.2)."""

from __future__ import annotations

import httpx
import pytest

from agents_core.tools.common.direct import (
    DIRECT_API_URL,
    DirectClient,
    DirectError,
    make_direct_tools,
)


def _transport(routes: dict[str, dict]) -> httpx.MockTransport:
    """routes: path-segment → JSON dict (or Exception) to return."""

    def handler(request: httpx.Request) -> httpx.Response:
        segment = request.url.path.split("/")[-1]
        if segment not in routes:
            err = {"error": {"error_code": 0, "error_string": "no_route"}}
            return httpx.Response(404, json=err)
        route = routes[segment]
        return httpx.Response(200, json=route)

    return httpx.MockTransport(handler)


class TestDirectClient:
    def test_requires_token(self):
        with pytest.raises(ValueError):
            DirectClient("")

    @pytest.mark.asyncio
    async def test_sets_bearer_and_lang(self):
        captured: dict = {}

        async def h(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json={"result": {"Campaigns": []}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        dc = DirectClient("tok", client=client)
        await dc.call("campaigns", "get", {"SelectionCriteria": {}, "FieldNames": []})
        assert captured["headers"]["authorization"] == "Bearer tok"
        assert captured["headers"]["accept-language"] == "ru"

    @pytest.mark.asyncio
    async def test_raises_on_api_error(self):
        err = {"campaigns": {"error": {"error_code": 54, "error_string": "No permissions"}}}
        client = httpx.AsyncClient(transport=_transport(err))
        dc = DirectClient("tok", client=client)
        with pytest.raises(DirectError) as ei:
            await dc.call("campaigns", "get", {})
        assert ei.value.code == 54


class TestFactoryShape:
    def test_8_tools_with_verify_pairs(self):
        dc = DirectClient("tok")
        tools = make_direct_tools(dc)
        names = [t.name for t in tools]
        assert names == [
            "direct_campaigns_get",
            "direct_campaign_pause",
            "direct_campaign_resume",
            "direct_campaign_update_budget",
            "direct_campaign_verify_state",
            "direct_keywords_get",
            "direct_keyword_update_bid",
            "direct_keyword_verify_bid",
        ]
        # Tiers
        tiers = {t.name: t.tier for t in tools}
        assert tiers["direct_campaigns_get"] == "read"
        assert tiers["direct_campaign_pause"] == "write"
        assert tiers["direct_campaign_resume"] == "write"
        assert tiers["direct_campaign_update_budget"] == "write"
        assert tiers["direct_campaign_verify_state"] == "read"
        assert tiers["direct_keywords_get"] == "read"
        assert tiers["direct_keyword_update_bid"] == "write"
        assert tiers["direct_keyword_verify_bid"] == "read"

    def test_write_tools_require_idempotency_key(self):
        dc = DirectClient("tok")
        for t in make_direct_tools(dc):
            if t.tier == "write":
                assert "idempotency_key" in t.input_schema["required"], t.name
                assert t.requires_verify is True, t.name


@pytest.mark.asyncio
class TestCampaignsGet:
    async def test_returns_campaign_list(self):
        client = httpx.AsyncClient(
            transport=_transport(
                {"campaigns": {"result": {"Campaigns": [{"Id": 1, "Name": "X", "State": "ON"}]}}}
            )
        )
        dc = DirectClient("tok", client=client)
        get = next(t for t in make_direct_tools(dc) if t.name == "direct_campaigns_get")
        result = await get.handler()
        assert result == {"Campaigns": [{"Id": 1, "Name": "X", "State": "ON"}]}

    async def test_filters_by_ids(self):
        captured: dict = {}

        async def h(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"result": {"Campaigns": []}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        dc = DirectClient("tok", client=client)
        get = next(t for t in make_direct_tools(dc) if t.name == "direct_campaigns_get")
        await get.handler(campaign_ids=[100, 200])
        assert '"Ids":[100,200]' in captured["body"]


@pytest.mark.asyncio
class TestPauseResume:
    async def test_pause_posts_suspend(self):
        captured: dict = {}

        async def h(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"result": {"SuspendResults": [{"Id": 100}]}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        dc = DirectClient("tok", client=client)
        pause = next(t for t in make_direct_tools(dc) if t.name == "direct_campaign_pause")
        await pause.handler(campaign_ids=[100], idempotency_key="k-1")
        assert '"method":"suspend"' in captured["body"]
        assert '"Ids":[100]' in captured["body"]

    async def test_resume_posts_resume(self):
        captured: dict = {}

        async def h(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"result": {"ResumeResults": []}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        dc = DirectClient("tok", client=client)
        res = next(t for t in make_direct_tools(dc) if t.name == "direct_campaign_resume")
        await res.handler(campaign_ids=[100, 200], idempotency_key="k-2")
        assert '"method":"resume"' in captured["body"]


@pytest.mark.asyncio
class TestBudgetUpdate:
    async def test_update_sends_micros(self):
        captured: dict = {}

        async def h(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"result": {"UpdateResults": []}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        dc = DirectClient("tok", client=client)
        upd = next(
            t for t in make_direct_tools(dc) if t.name == "direct_campaign_update_budget"
        )
        await upd.handler(
            campaign_id=100, daily_budget_rub=5000, mode="STANDARD", idempotency_key="k",
        )
        # 5000 rub -> 5_000_000_000 micros
        assert '"Amount":5000000000' in captured["body"]
        assert '"Mode":"STANDARD"' in captured["body"]

    async def test_invalid_mode_rejected(self):
        dc = DirectClient("tok")
        upd = next(
            t for t in make_direct_tools(dc) if t.name == "direct_campaign_update_budget"
        )
        with pytest.raises(ValueError):
            await upd.handler(
                campaign_id=100, daily_budget_rub=1000, mode="BOGUS", idempotency_key="k",
            )


@pytest.mark.asyncio
class TestCampaignVerifyState:
    async def test_verify_matches(self):
        client = httpx.AsyncClient(
            transport=_transport(
                {"campaigns": {"result": {"Campaigns": [{"Id": 100, "State": "SUSPENDED"}]}}}
            )
        )
        dc = DirectClient("tok", client=client)
        vf = next(
            t for t in make_direct_tools(dc) if t.name == "direct_campaign_verify_state"
        )
        result = await vf.handler(campaign_id=100, expected_state="SUSPENDED")
        assert result == {
            "verified": True,
            "actual_state": "SUSPENDED",
            "expected_state": "SUSPENDED",
        }

    async def test_verify_mismatch(self):
        client = httpx.AsyncClient(
            transport=_transport(
                {"campaigns": {"result": {"Campaigns": [{"Id": 100, "State": "ON"}]}}}
            )
        )
        dc = DirectClient("tok", client=client)
        vf = next(
            t for t in make_direct_tools(dc) if t.name == "direct_campaign_verify_state"
        )
        result = await vf.handler(campaign_id=100, expected_state="SUSPENDED")
        assert result["verified"] is False
        assert result["actual_state"] == "ON"


@pytest.mark.asyncio
class TestKeywords:
    async def test_keywords_get(self):
        client = httpx.AsyncClient(
            transport=_transport(
                {"keywords": {"result": {"Keywords": [{"Id": 5, "Keyword": "банкротство"}]}}}
            )
        )
        dc = DirectClient("tok", client=client)
        get = next(t for t in make_direct_tools(dc) if t.name == "direct_keywords_get")
        result = await get.handler(ad_group_id=42)
        assert result["Keywords"][0]["Id"] == 5

    async def test_update_bid_sends_micros(self):
        captured: dict = {}

        async def h(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"result": {"SetResults": []}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        dc = DirectClient("tok", client=client)
        upd = next(
            t for t in make_direct_tools(dc) if t.name == "direct_keyword_update_bid"
        )
        await upd.handler(keyword_id=5, bid_rub=12.50, idempotency_key="k")
        # 12.50 -> 12_500_000 micros
        assert '"SearchBid":12500000' in captured["body"]
        assert '"KeywordId":5' in captured["body"]

    async def test_verify_bid_within_tolerance(self):
        ok = {"keywordbids": {"result": {"KeywordBids": [
            {"KeywordId": 5, "SearchBid": 12_505_000}
        ]}}}
        client = httpx.AsyncClient(transport=_transport(ok))
        dc = DirectClient("tok", client=client)
        vf = next(
            t for t in make_direct_tools(dc) if t.name == "direct_keyword_verify_bid"
        )
        result = await vf.handler(keyword_id=5, expected_bid_rub=12.5, tolerance_rub=0.01)
        assert result["verified"] is True
        assert result["actual_bid_rub"] == 12.51

    async def test_verify_bid_outside_tolerance(self):
        off = {"keywordbids": {"result": {"KeywordBids": [
            {"KeywordId": 5, "SearchBid": 15_000_000}
        ]}}}
        client = httpx.AsyncClient(transport=_transport(off))
        dc = DirectClient("tok", client=client)
        vf = next(
            t for t in make_direct_tools(dc) if t.name == "direct_keyword_verify_bid"
        )
        result = await vf.handler(keyword_id=5, expected_bid_rub=12.5, tolerance_rub=0.01)
        assert result["verified"] is False
        assert result["actual_bid_rub"] == 15.0


def test_api_url_constant():
    assert DIRECT_API_URL.startswith("https://api.direct.yandex.com")
