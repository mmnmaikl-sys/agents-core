"""Task 0.21 DoD — real smoke against live RAG service.

Runs `rag_search` against the production RAG on Railway with the master
RAG_API_KEY. Expects at least one chunk back for a known project query.

Usage:
    RAG_API_KEY=... python tests/integration_rag_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from agents_core.tools.common.rag import make_rag_tools


async def main() -> int:
    key = os.environ.get("RAG_API_KEY")
    if not key:
        print("[rag-smoke] RAG_API_KEY missing", file=sys.stderr)
        return 2

    search = next(t for t in make_rag_tools(key) if t.name == "rag_search")
    result = await search.handler(question="как работают звонки в ОКК", top_k=3)

    # /search returns {chunks: [{source_type, source_id, content, ...}]};
    # /ask adds answer + cost_usd on top.
    chunks = result.get("chunks") or result.get("sources") or []
    print(f"[rag-smoke] chunks={len(chunks) if isinstance(chunks, list) else 'n/a'}")
    if isinstance(chunks, list) and chunks:
        first = chunks[0]
        print(f"[rag-smoke] first source_type={first.get('source_type')!r} "
              f"source_id={first.get('source_id')!r}")

    assert isinstance(result, dict), f"response not a dict: {type(result)}"
    assert "chunks" in result or "sources" in result, f"unexpected keys: {list(result)}"
    assert isinstance(chunks, list) and len(chunks) > 0, "zero chunks for a known query"
    print("[rag-smoke] OK — endpoint answered with expected shape")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
