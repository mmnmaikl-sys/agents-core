"""Unit tests for agents_core.tools.common.telegram (task 0.20)."""

from __future__ import annotations

import json

import httpx
import pytest

from agents_core.tools.common.telegram import (
    TelegramClient,
    TelegramError,
    make_telegram_tools,
)


@pytest.mark.asyncio
class TestTelegramClient:
    async def test_rejects_empty_token(self):
        with pytest.raises(ValueError):
            TelegramClient("")

    async def test_raises_on_not_ok(self):
        async def h(_req: httpx.Request):
            return httpx.Response(
                200, json={"ok": False, "error_code": 400, "description": "bad"}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        tg = TelegramClient("123:abc", client=client)
        with pytest.raises(TelegramError):
            await tg.call("sendMessage", {"chat_id": 1, "text": "hi"})


@pytest.mark.asyncio
class TestFactoryShape:
    async def test_three_tools(self):
        tg = TelegramClient("123:abc")
        names = [t.name for t in make_telegram_tools(tg)]
        assert names == ["tg_send_message", "tg_send_document", "tg_edit_message"]


@pytest.mark.asyncio
class TestSendMessage:
    async def test_happy_path(self):
        captured: dict = {}

        async def h(request: httpx.Request):
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(
                200,
                json={"ok": True, "result": {"message_id": 42, "chat": {"id": 1}}},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        tg = TelegramClient("t:t", client=client)
        send = next(t for t in make_telegram_tools(tg) if t.name == "tg_send_message")
        result = await send.handler(
            chat_id=1, text="hello", idempotency_key="k1", parse_mode="Markdown"
        )
        assert result["message_id"] == 42
        assert captured["body"]["chat_id"] == 1
        assert "[idem:k1]" in captured["body"]["text"]
        assert captured["body"]["parse_mode"] == "Markdown"
        assert captured["body"]["disable_web_page_preview"] is True


@pytest.mark.asyncio
class TestEditMessage:
    async def test_edit_is_idempotent_by_design(self):
        async def h(_req):
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 9}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        tg = TelegramClient("t:t", client=client)
        edit = next(t for t in make_telegram_tools(tg) if t.name == "tg_edit_message")
        assert edit.idempotent is True
        await edit.handler(chat_id=1, message_id=9, text="upd")


@pytest.mark.asyncio
class TestSendDocument:
    async def test_url_mode(self):
        captured: dict = {}

        async def h(request: httpx.Request):
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(200, json={"ok": True, "result": {}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        tg = TelegramClient("t:t", client=client)
        doc = next(t for t in make_telegram_tools(tg) if t.name == "tg_send_document")
        await doc.handler(
            chat_id=1,
            document_url="https://x.example/f.pdf",
            caption="отчёт",
            idempotency_key="k1",
        )
        assert captured["body"]["document"] == "https://x.example/f.pdf"
        assert "[idem:k1]" in captured["body"]["caption"]
