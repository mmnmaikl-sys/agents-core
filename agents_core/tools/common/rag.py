"""RAG Service tools (task 0.21).

Thin wrapper over the internal RAG Service
(``https://rag-service-production-a41d.up.railway.app``).

Two tools:
- ``rag_search`` → ``POST /search`` (semantic + pg_trgm hybrid, no LLM).
- ``rag_upload`` → ``POST /upload`` (ingest a chunk).

The service contracts use ``question`` + ``top_k`` + optional ``filters``
(NOT ``query`` / ``k`` / ``domain``). See
``work/rag-service/app/server.py::AskRequest`` and ``AskFilters`` for the
authoritative schema.

Domain filtering is done server-side via the per-agent X-API-Key (each
key is bound to ``allowed_domains`` in the ``agent_profiles`` table); the
client doesn't pass ``agent_name`` in the body. Instead the caller passes
the key for the right agent.
"""

from __future__ import annotations

from typing import Any

import httpx

from agents_core.tools.registry import Tool

__all__ = ["DEFAULT_RAG_BASE_URL", "make_rag_tools"]

DEFAULT_RAG_BASE_URL = "https://rag-service-production-a41d.up.railway.app"


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float = 30.0,
) -> dict[str, Any]:
    resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def make_rag_tools(
    api_key: str,
    base_url: str = DEFAULT_RAG_BASE_URL,
    client: httpx.AsyncClient | None = None,
) -> list[Tool]:
    """Return ``[rag_search, rag_upload]`` bound to a per-agent X-API-Key.

    Parameters
    ----------
    api_key:
        Per-agent X-API-Key (see ``rag-service/app/middleware`` for the
        auth layer). The same key determines which domains the agent can
        read — don't reuse the master key across agents.
    base_url:
        Overrides the default Railway host for dev/preview.
    client:
        Optional pre-built ``AsyncClient`` for connection pooling.
    """
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    async def _with_client(fn):
        """Run fn(client) reusing external client or opening a fresh one."""
        if client is not None:
            return await fn(client)
        async with httpx.AsyncClient() as c:
            return await fn(c)

    async def _search(
        question: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ):
        payload: dict[str, Any] = {"question": question, "top_k": top_k}
        if filters:
            payload["filters"] = filters
        return await _with_client(
            lambda c: _post_json(c, f"{base_url}/search", headers, payload)
        )

    async def _upload(
        name: str,
        content: str,
        source_type: str,
        domain: str,
        idempotency_key: str,
    ):
        payload = {
            "name": name,
            "content": f"[idem:{idempotency_key}]\n{content}",
            "source_type": source_type,
            "domain": domain,
        }
        return await _with_client(
            lambda c: _post_json(c, f"{base_url}/upload", headers, payload, timeout=60.0)
        )

    return [
        Tool(
            name="rag_search",
            description=(
                "Hybrid search over the RAG knowledge base (pgvector + pg_trgm). "
                "Returns top-k chunks with text + metadata. No LLM call, cheap."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Optional AskFilters: source_type, "
                            "exclude_source_types[], operator, date_from."
                        ),
                    },
                },
                "required": ["question"],
            },
            handler=_search,
            tier="read",
            idempotent=True,
            tags=("rag", "read"),
        ),
        Tool(
            name="rag_upload",
            description=(
                "Upload a document chunk to the RAG index. Use when new "
                "institutional knowledge needs to be persisted for agents."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                    "source_type": {"type": "string"},
                    "domain": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": [
                    "name",
                    "content",
                    "source_type",
                    "domain",
                    "idempotency_key",
                ],
            },
            handler=_upload,
            tier="write",
            idempotent=False,
            requires_verify=False,  # upload endpoint has its own dedup
            tags=("rag", "write"),
        ),
    ]
