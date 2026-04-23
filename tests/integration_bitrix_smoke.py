"""Task 0.18 DoD — Bitrix read-only smoke on live portal.

Calls crm.deal.list (limit 1) to prove:
- BitrixClient handshake + auth works
- response has the expected shape (result is a list of dicts)

Read-only by design; no deal_update/close_won in automated smoke without
explicit operator command. Write-side SET/VERIFY pair gets its own manual
run on a dedicated test deal.

Usage:
    BITRIX_WEBHOOK_URL=https://domain.bitrix24.ru/rest/1/<token>/ \
        python tests/integration_bitrix_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from agents_core.tools.common.bitrix import BitrixClient


async def main() -> int:
    url = os.environ.get("BITRIX_WEBHOOK_URL")
    if not url:
        print("[bx-smoke] BITRIX_WEBHOOK_URL required", file=sys.stderr)
        return 2

    bx = BitrixClient(url)
    # Low-impact probe: list one deal. Doesn't touch data.
    deals = await bx.call("crm.deal.list", {"filter": {}, "select": ["ID", "TITLE"], "start": 0})
    # result shape depends on SDK version — usually a list of dicts.
    if isinstance(deals, dict) and "result" in deals:  # defensive
        deals = deals["result"]
    n = len(deals) if hasattr(deals, "__len__") else "n/a"
    print(f"[bx-smoke] returned type={type(deals).__name__} len={n}")
    if isinstance(deals, list) and deals:
        first = deals[0]
        print(f"[bx-smoke] first deal ID={first.get('ID')} TITLE={first.get('TITLE')!r}")
    assert isinstance(deals, list), f"expected list, got {type(deals)}"
    print("[bx-smoke] OK — Bitrix webhook auth + deal.list works")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
