"""Task 0.22 DoD — real HH.ru smoke with dev access token.

Calls hh_get_resumes against the live HH API. Requires a valid bearer
token (from hr-director Railway env or memory/hh_api_credentials.md).

Usage:
    HH_ACCESS_TOKEN=... python tests/integration_hh_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from agents_core.tools.common.hh import HHClient, make_hh_tools


async def main() -> int:
    token = os.environ.get("HH_ACCESS_TOKEN")
    if not token:
        print("[hh-smoke] HH_ACCESS_TOKEN required", file=sys.stderr)
        return 2

    hh = HHClient(token, user_agent="agents-core-smoke (myaklov@yandex.ru)")
    tools = {t.name: t for t in make_hh_tools(hh)}

    resumes = await tools["hh_get_resumes"].handler(text="юрист", area=88, per_page=3)
    items = resumes.get("items", [])
    print(f"[hh-smoke] resumes returned: {len(items)}")
    if items:
        first = items[0]
        title = first.get("title", "?")
        area = first.get("area", {}).get("name", "?")
        print(f"[hh-smoke] first title={title!r} area={area}")
    assert isinstance(items, list), f"expected list, got {type(items)}"
    print("[hh-smoke] OK — HH API auth + resumes search works")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
