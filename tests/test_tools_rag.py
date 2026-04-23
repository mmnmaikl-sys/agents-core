"""Unit tests for agents_core.tools.common.rag (task 0.21)."""

from __future__ import annotations

import httpx
import pytest

from agents_core.tools.common.rag import DEFAULT_RAG_BASE_URL, make_rag_tools


class TestFactoryShape:
    def test_returns_two_tools(self):
        tools = make_rag_tools(api_key="k", agent_name="ceo")
        names = [t.name for t in tools]
        assert names == ["rag_search", "rag_upload"]

    def test_search_is_read_upload_is_write(self):
        tools = make_rag_tools(api_key="k", agent_name="ceo")
        assert tools[0].tier == "read"
        assert tools[1].tier == "write"

    def test_upload_requires_idempotency_key_in_schema(self):
        upload = [t for t in make_rag_tools("k", "ceo") if t.name == "rag_upload"][0]
        assert "idempotency_key" in upload.input_schema["required"]


@pytest.mark.asyncio
class TestSearch:
    async def test_posts_search_payload_with_agent_name(self):
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"results": [{"text": "hit"}]})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        search = next(
            t
            for t in make_rag_tools("secret", "ceo", client=client)
            if t.name == "rag_search"
        )
        result = await search.handler(query="как работают звонки", k=3)
        assert result == {"results": [{"text": "hit"}]}
        assert f"{DEFAULT_RAG_BASE_URL}/search" == captured["url"]
        assert captured["headers"]["x-api-key"] == "secret"
        assert '"agent_name":"ceo"' in captured["body"]
        assert '"k":3' in captured["body"]

    async def test_custom_base_url_respected(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            assert "staging.rag.example" in str(request.url)
            return httpx.Response(200, json={"results": []})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        search = next(
            t
            for t in make_rag_tools(
                "k", "ceo", base_url="https://staging.rag.example", client=client
            )
            if t.name == "rag_search"
        )
        await search.handler(query="test")

    async def test_raises_on_http_error(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"detail": "forbidden"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        search = next(
            t for t in make_rag_tools("bad", "x", client=client) if t.name == "rag_search"
        )
        with pytest.raises(httpx.HTTPStatusError):
            await search.handler(query="test")


@pytest.mark.asyncio
class TestUpload:
    async def test_upload_posts_payload(self):
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = request.content.decode()
            return httpx.Response(200, json={"id": "doc-1", "chunks": 3})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        upload = next(
            t for t in make_rag_tools("k", "hr", client=client) if t.name == "rag_upload"
        )
        result = await upload.handler(
            text="Важное правило...",
            title="Лучшие практики",
            domain="hr",
            idempotency_key="k-1",
            metadata={"source": "manual"},
        )
        assert result == {"id": "doc-1", "chunks": 3}
        assert "/upload" in captured["url"]
        assert '"domain":"hr"' in captured["body"]
        assert '"source":"manual"' in captured["body"]
