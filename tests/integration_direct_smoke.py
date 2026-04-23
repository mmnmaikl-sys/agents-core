"""Task 1.2 DoD — real Direct API smoke.

Calls direct_campaigns_get to confirm auth + endpoint shape without
mutating anything. Read-only by design.

Usage:
    DIRECT_API_TOKEN=... python tests/integration_direct_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from agents_core.tools.common.direct import DirectClient, make_direct_tools


async def main() -> int:
    token = os.environ.get("DIRECT_API_TOKEN")
    if not token:
        print("[direct-smoke] DIRECT_API_TOKEN required", file=sys.stderr)
        return 2

    dc = DirectClient(token)
    get = next(t for t in make_direct_tools(dc) if t.name == "direct_campaigns_get")
    result = await get.handler()
    campaigns = result.get("Campaigns", [])
    print(f"[direct-smoke] campaigns returned: {len(campaigns)}")
    if campaigns:
        for c in campaigns[:5]:
            name = c.get("Name", "?")[:60]
            st = c.get("State", "?")
            print(f"[direct-smoke]   Id={c.get('Id')} State={st:<10} Name={name}")
    assert isinstance(campaigns, list), f"expected list, got {type(campaigns)}"
    print("[direct-smoke] OK — Direct v5 auth + campaigns.get works")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
