"""Task 0.20 DoD — real Telegram smoke.

Sends one message to Mikhail's chat via the ai-cmo bot (known-live token,
unified-bot's token was revoked per ai_cmo_orchestrator_deployed.md).

Usage:
    TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... \
        python tests/integration_telegram_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from agents_core.tools.common.telegram import TelegramClient, make_telegram_tools


async def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[tg-smoke] TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required", file=sys.stderr)
        return 2

    tg = TelegramClient(token)
    send = next(t for t in make_telegram_tools(tg) if t.name == "tg_send_message")
    res = await send.handler(
        chat_id=int(chat_id),
        text="agents-core smoke (task 0.20) — можно игнорить",
        idempotency_key="agents-core-smoke-0.20",
    )
    print(f"[tg-smoke] message_id={res['message_id']} chat_id={res['chat']['id']}")
    assert "message_id" in res and res["message_id"] > 0
    print("[tg-smoke] OK — sendMessage accepted")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
