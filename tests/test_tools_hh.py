"""Unit tests for agents_core.tools.common.hh (task 0.22)."""

from __future__ import annotations

import httpx
import pytest

from agents_core.tools.common.hh import HHClient, make_hh_tools


class TestConstructor:
    def test_requires_access_token(self):
        with pytest.raises(ValueError):
            HHClient("", user_agent="a (b@c)")

    def test_requires_user_agent(self):
        with pytest.raises(ValueError):
            HHClient("tok", user_agent="")


class TestFactoryShape:
    def test_three_tools(self):
        hh = HHClient("tok", user_agent="a (b@c)")
        names = [t.name for t in make_hh_tools(hh)]
        assert names == ["hh_get_resumes", "hh_get_applications", "hh_send_invitation"]

    def test_send_invitation_requires_idempotency_key(self):
        hh = HHClient("tok", user_agent="a (b@c)")
        inv = next(t for t in make_hh_tools(hh) if t.name == "hh_send_invitation")
        assert "idempotency_key" in inv.input_schema["required"]


@pytest.mark.asyncio
class TestGetResumes:
    async def test_sends_auth_and_useragent_headers(self):
        captured: dict = {}

        async def h(request: httpx.Request):
            captured["headers"] = dict(request.headers)
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"items": [], "pages": 0})

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        hh = HHClient("tok", user_agent="MyApp (me@example.com)", client=client)
        tool = next(t for t in make_hh_tools(hh) if t.name == "hh_get_resumes")
        await tool.handler(text="python", area=88, per_page=10, page=0)
        assert captured["headers"]["authorization"] == "Bearer tok"
        assert captured["headers"]["hh-user-agent"] == "MyApp (me@example.com)"
        assert "text=python" in captured["url"]
        assert "area=88" in captured["url"]


@pytest.mark.asyncio
class TestGetApplications:
    async def test_vacancy_id_param(self):
        captured: dict = {}

        async def h(request: httpx.Request):
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"items": [], "pages": 0})

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        hh = HHClient("tok", user_agent="a (b@c)", client=client)
        tool = next(t for t in make_hh_tools(hh) if t.name == "hh_get_applications")
        await tool.handler(vacancy_id=12345)
        assert "vacancy_id=12345" in captured["url"]
        assert "/negotiations" in captured["url"]


@pytest.mark.asyncio
class TestSendInvitation:
    async def test_stashes_idempotency_key_in_message(self):
        captured: dict = {}

        async def h(request: httpx.Request):
            captured["body"] = request.content.decode()
            return httpx.Response(201, json={"id": "neg-1"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        hh = HHClient("tok", user_agent="a (b@c)", client=client)
        tool = next(t for t in make_hh_tools(hh) if t.name == "hh_send_invitation")
        res = await tool.handler(
            resume_id="r-1",
            vacancy_id=100,
            message="Приглашаем",
            idempotency_key="abc",
        )
        assert res == {"id": "neg-1"}
        assert "[idem:abc]" in captured["body"]
