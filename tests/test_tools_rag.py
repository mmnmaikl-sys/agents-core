"""Unit tests for agents_core.tools.common.rag (task 0.21).

Real contract (see work/rag-service/app/server.py::AskRequest):
  POST /search  {question: str, top_k: int, filters?: {...}}
  POST /upload  {name, content, source_type, domain}

Domain filtering is done server-side via the per-agent X-API-Key, not the body.
"""

from __future__ import annotations

import httpx
import pytest

from agents_core.tools.common.rag import DEFAULT_RAG_BASE_URL, make_rag_tools


class TestFactoryShape:
    def test_returns_two_tools(self):
        tools = make_rag_tools(api_key="k")
        assert [t.name for t in tools] == ["rag_search", "rag_upload"]

    def test_search_is_read_upload_is_write(self):
        tools = make_rag_tools(api_key="k")
        assert tools[0].tier == "read"
        assert tools[1].tier == "write"

    def test_upload_requires_idempotency_key_in_schema(self):
        upload = [t for t in make_rag_tools("k") if t.name == "rag_upload"][0]
        assert "idempotency_key" in upload.input_schema["required"]

    def test_search_schema_uses_question_top_k(self):
        search = [t for t in make_rag_tools("k") if t.name == "rag_search"][0]
        assert "question" in search.input_schema["required"]
        assert "question" in search.input_schema["properties"]
        assert "top_k" in search.input_schema["properties"]


@pytest.mark.asyncio
class TestSearch:
    async def test_posts_question_and_top_k(self):
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"answer": "", "sources": [], "cost_usd": 0})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        search = next(
            t for t in make_rag_tools("secret", client=client) if t.name == "rag_search"
        )
        await search.handler(question="как работают звонки", top_k=3)
        assert f"{DEFAULT_RAG_BASE_URL}/search" == captured["url"]
        assert captured["headers"]["x-api-key"] == "secret"
        assert '"question":"как работают звонки"' in captured["body"]
        assert '"top_k":3' in captured["body"]
        # agent_name is NOT in the body (server uses API key for domain filter)
        assert "agent_name" not in captured["body"]

    async def test_filters_are_passed_through(self):
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        search = next(
            t for t in make_rag_tools("k", client=client) if t.name == "rag_search"
        )
        await search.handler(
            question="q", filters={"source_type": "call", "operator": "ivanov"}
        )
        assert '"filters"' in captured["body"]
        assert '"operator":"ivanov"' in captured["body"]

    async def test_custom_base_url_respected(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            assert "staging.rag.example" in str(request.url)
            return httpx.Response(200, json={})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        search = next(
            t
            for t in make_rag_tools(
                "k", base_url="https://staging.rag.example", client=client
            )
            if t.name == "rag_search"
        )
        await search.handler(question="test")

    async def test_raises_on_http_error(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"detail": "forbidden"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        search = next(
            t for t in make_rag_tools("bad", client=client) if t.name == "rag_search"
        )
        with pytest.raises(httpx.HTTPStatusError):
            await search.handler(question="test")


@pytest.mark.asyncio
class TestUpload:
    async def test_upload_sends_server_fields(self):
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"chunks": 3})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        upload = next(
            t for t in make_rag_tools("k", client=client) if t.name == "rag_upload"
        )
        result = await upload.handler(
            name="Лучшие практики",
            content="Важное правило...",
            source_type="knowledge",
            domain="hr",
            idempotency_key="k-1",
        )
        assert result == {"chunks": 3}
        assert "/upload" in captured["url"]
        assert '"name":"Лучшие практики"' in captured["body"]
        assert '"source_type":"knowledge"' in captured["body"]
        assert '"domain":"hr"' in captured["body"]
        assert "[idem:k-1]" in captured["body"]
