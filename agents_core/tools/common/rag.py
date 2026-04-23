"""RAG Service tools (task 0.21).

Thin wrapper over the internal RAG Service (pgvector + DeepSeek embeddings):
- ``rag_search`` — semantic search across the knowledge base.
- ``rag_upload`` — ingest a new document chunk.

Both talk to the Railway-hosted service over HTTPS with an ``X-API-Key``
header.

Default base URL matches the deployed instance today; per-env override via
``base_url=`` keeps dev/prod separation clean. Real integration smoke
against ``rag-service-production-a41d.up.railway.app`` is the task-0.21 DoD
and is deferred to the post-session smoke run.
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
    agent_name: str,
    base_url: str = DEFAULT_RAG_BASE_URL,
    client: httpx.AsyncClient | None = None,
) -> list[Tool]:
    """Return ``[rag_search, rag_upload]`` bound to ``agent_name``.

    Parameters
    ----------
    api_key:
        X-API-Key value. Per-agent keys are the norm (see
        ``memory/reference_credentials_and_services.md``).
    agent_name:
        Required so the RAG service filters to the agent's domain; without
        it the service returns cross-agent content and the answers get
        noisy (``feedback_check_existing_first`` — "RAG fallback требует
        agent_name").
    client:
        Pass a pre-built ``AsyncClient`` to share connections across calls.
        A fresh one is created per tool-call otherwise.
    """
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    async def _search(query: str, k: int = 5, domain: str | None = None):
        async with (client or httpx.AsyncClient()) as c:
            payload: dict[str, Any] = {
                "query": query,
                "k": k,
                "agent_name": agent_name,
            }
            if domain:
                payload["domain"] = domain
            return await _post_json(c, f"{base_url}/search", headers, payload)

    async def _upload(
        text: str,
        title: str,
        domain: str,
        metadata: dict[str, Any] | None = None,
    ):
        async with (client or httpx.AsyncClient()) as c:
            payload = {
                "text": text,
                "title": title,
                "domain": domain,
                "agent_name": agent_name,
                "metadata": metadata or {},
            }
            return await _post_json(
                c, f"{base_url}/upload", headers, payload, timeout=60.0
            )

    return [
        Tool(
            name="rag_search",
            description=(
                "Semantic search over the RAG knowledge base. Returns top-k "
                "chunks with text + metadata. Use when the answer requires "
                "grounded context from project knowledge."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                    "domain": {
                        "type": "string",
                        "description": "Optional domain filter, e.g. 'okk', 'crm'.",
                    },
                },
                "required": ["query"],
            },
            handler=_search,
            tier="read",
            idempotent=True,
            tags=("rag", "read"),
        ),
        Tool(
            name="rag_upload",
            description=(
                "Upload a document chunk to the RAG index. Use for "
                "persisting new institutional knowledge."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "title": {"type": "string"},
                    "domain": {"type": "string"},
                    "metadata": {"type": "object"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["text", "title", "domain", "idempotency_key"],
            },
            handler=lambda text, title, domain, idempotency_key, metadata=None: _upload(
                text, title, domain, metadata
            ),
            tier="write",
            idempotent=False,
            requires_verify=False,  # upload endpoint has its own dedup
            tags=("rag", "write"),
        ),
    ]
