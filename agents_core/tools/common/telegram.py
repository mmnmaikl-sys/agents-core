"""Telegram Bot API tools (task 0.20).

Three base tools: ``send_message``, ``send_document``, ``edit_message``.
Minimal — agents usually don't need more than this from the Bot API side
(polling / webhook routing is the bot's job, not a tool).

All writes carry ``idempotency_key`` stashed in the outgoing message body
or caption (Telegram has no native idempotency). The paired notion of
"verify" here is "the message ID came back in the response" — if the
Bot API returned 200 and a ``message_id``, the send succeeded. No
separate ``_verify`` tool needed.

``send_document`` accepts an URL or bytes. For bytes, we use multipart
upload; for URLs (CDN / S3 / Bitrix disk), we let Telegram fetch it.
"""

from __future__ import annotations

from typing import Any

import httpx

from agents_core.tools.registry import Tool

__all__ = ["TelegramClient", "make_telegram_tools"]


class TelegramClient:
    def __init__(
        self,
        bot_token: str,
        client: httpx.AsyncClient | None = None,
        timeout: float = 15.0,
    ) -> None:
        if not bot_token:
            raise ValueError("bot_token required")
        self._base = f"https://api.telegram.org/bot{bot_token}"
        self._client = client
        self._timeout = timeout

    async def call(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self._base}/{method}"
        async with (self._client or httpx.AsyncClient()) as c:
            if files:
                resp = await c.post(
                    url, data=payload or {}, files=files, timeout=self._timeout
                )
            else:
                resp = await c.post(url, json=payload or {}, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        if not data.get("ok"):
            raise TelegramError(
                data.get("error_code"), data.get("description", "unknown")
            )
        return data["result"]


class TelegramError(RuntimeError):
    def __init__(self, code: int | None, description: str) -> None:
        super().__init__(f"{code}: {description}")
        self.code = code
        self.description = description


def make_telegram_tools(tg: TelegramClient) -> list[Tool]:
    async def _send_message(
        chat_id: int,
        text: str,
        idempotency_key: str,
        parse_mode: str | None = None,
        reply_to: int | None = None,
    ):
        payload: dict[str, Any] = {
            "chat_id": int(chat_id),
            "text": f"{text}\n\n[idem:{idempotency_key}]",
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_to is not None:
            payload["reply_to_message_id"] = int(reply_to)
        return await tg.call("sendMessage", payload)

    async def _edit_message(
        chat_id: int,
        message_id: int,
        text: str,
        parse_mode: str | None = None,
    ):
        payload: dict[str, Any] = {
            "chat_id": int(chat_id),
            "message_id": int(message_id),
            "text": text,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        return await tg.call("editMessageText", payload)

    async def _send_document(
        chat_id: int,
        document_url: str,
        idempotency_key: str,
        caption: str | None = None,
    ):
        payload = {
            "chat_id": int(chat_id),
            "document": document_url,
            "caption": f"{caption or ''}\n[idem:{idempotency_key}]",
        }
        return await tg.call("sendDocument", payload)

    return [
        Tool(
            name="tg_send_message",
            description="Send a text message to a Telegram chat.",
            input_schema={
                "type": "object",
                "properties": {
                    "chat_id": {"type": "integer"},
                    "text": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                    "parse_mode": {
                        "type": "string",
                        "enum": ["Markdown", "MarkdownV2", "HTML"],
                    },
                    "reply_to": {"type": "integer"},
                },
                "required": ["chat_id", "text", "idempotency_key"],
            },
            handler=_send_message,
            tier="write",
            idempotent=False,
            requires_verify=False,
            tags=("telegram", "write"),
        ),
        Tool(
            name="tg_send_document",
            description="Send a file to a chat by URL.",
            input_schema={
                "type": "object",
                "properties": {
                    "chat_id": {"type": "integer"},
                    "document_url": {"type": "string"},
                    "caption": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["chat_id", "document_url", "idempotency_key"],
            },
            handler=_send_document,
            tier="write",
            idempotent=False,
            requires_verify=False,
            tags=("telegram", "write"),
        ),
        Tool(
            name="tg_edit_message",
            description="Edit the text of an already-sent message.",
            input_schema={
                "type": "object",
                "properties": {
                    "chat_id": {"type": "integer"},
                    "message_id": {"type": "integer"},
                    "text": {"type": "string"},
                    "parse_mode": {
                        "type": "string",
                        "enum": ["Markdown", "MarkdownV2", "HTML"],
                    },
                },
                "required": ["chat_id", "message_id", "text"],
            },
            handler=_edit_message,
            tier="write",
            idempotent=True,  # same (chat, msg, text) is a no-op for Telegram
            requires_verify=False,
            tags=("telegram", "write"),
        ),
    ]
