"""Unit tests for agents_core.llm.client.

Mocks Anthropic AsyncAnthropic and httpx to avoid real network calls.
Coverage goals:
- happy path (Anthropic) — usage, cost, cache tokens, Langfuse span lifecycle
- retry on APIError → success on 2nd attempt
- retries exhausted → exception surfaced, Langfuse span closed as ERROR
- cache_control injected when system_cache=True
- DeepSeek happy path via mocked httpx
- DeepSeek auto-routing via model="deepseek"
- chat_structured returns pydantic model instance
- no anthropic_api_key → RuntimeError on Anthropic call
- no deepseek_api_key → RuntimeError on DeepSeek call
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agents_core.llm.client import LLMClient, _system_with_cache


# --------------------------------------------------------------------- helpers
def _fake_anthropic_msg(
    text: str = "hello",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> SimpleNamespace:
    content = [SimpleNamespace(text=text)]
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )
    return SimpleNamespace(content=content, usage=usage)


def _patch_lf(monkeypatch) -> dict[str, MagicMock]:
    """Replace lf.start_generation / end_generation_* with MagicMocks."""
    calls: dict[str, MagicMock] = {
        "start": MagicMock(return_value=MagicMock(name="gen_handle")),
        "end_ok": MagicMock(),
        "end_err": MagicMock(),
    }
    from agents_core.llm import client as mod
    monkeypatch.setattr(mod.lf, "start_generation", calls["start"])
    monkeypatch.setattr(mod.lf, "end_generation_ok", calls["end_ok"])
    monkeypatch.setattr(mod.lf, "end_generation_err", calls["end_err"])
    return calls


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def client() -> LLMClient:
    c = LLMClient(anthropic_api_key="sk-test", deepseek_api_key="ds-test", backoff_base=1)
    c._anthropic = MagicMock()
    c._anthropic.messages = MagicMock()
    c._anthropic.messages.create = AsyncMock()
    return c


# ---------------------------------------------------------- system_with_cache
def test_system_with_cache_off():
    assert _system_with_cache("hello", False) == "hello"


def test_system_with_cache_on():
    out = _system_with_cache("hello", True)
    assert out == [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]


# ---------------------------------------------------------- Anthropic happy
@pytest.mark.asyncio
async def test_chat_anthropic_happy(client: LLMClient, monkeypatch):
    lf_calls = _patch_lf(monkeypatch)
    client._anthropic.messages.create.return_value = _fake_anthropic_msg(
        text="привет", input_tokens=100, output_tokens=20, cache_creation=64, cache_read=0,
    )

    resp = await client.chat(
        prompt="q", model="haiku", system="sys", system_cache=True,
        max_tokens=512, name="myagent.step1",
    )

    assert resp.text == "привет"
    assert resp.model == "claude-haiku-4-5-20251001"
    assert resp.usage.input == 100
    assert resp.usage.output == 20
    assert resp.usage.cache_creation == 64
    assert resp.usage.cache_read == 0
    # cost: (100*0.80 + 20*4.00) / 1e6 = 160 / 1e6
    assert resp.cost_usd == pytest.approx(160 / 1_000_000)
    assert resp.attempt == 1

    # cache_control should be in outgoing kwargs
    call_kwargs = client._anthropic.messages.create.call_args.kwargs
    assert isinstance(call_kwargs["system"], list)
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    # Langfuse span lifecycle
    assert lf_calls["start"].call_count == 1
    assert lf_calls["start"].call_args.kwargs["name"] == "myagent.step1"
    assert lf_calls["end_ok"].call_count == 1
    ok_kwargs = lf_calls["end_ok"].call_args.kwargs
    assert ok_kwargs["cache_creation_input_tokens"] == 64
    assert ok_kwargs["attempt"] == 1
    lf_calls["end_err"].assert_not_called()


# ---------------------------------------------------------- Anthropic retry
@pytest.mark.asyncio
async def test_chat_anthropic_retry_then_success(client: LLMClient, monkeypatch):
    _patch_lf(monkeypatch)
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    client._anthropic.messages.create.side_effect = [
        RuntimeError("overloaded"),
        _fake_anthropic_msg(text="recovered"),
    ]

    resp = await client.chat(prompt="q", model="haiku")
    assert resp.text == "recovered"
    assert resp.attempt == 2
    assert client._anthropic.messages.create.call_count == 2


@pytest.mark.asyncio
async def test_chat_anthropic_retries_exhausted(client: LLMClient, monkeypatch):
    lf_calls = _patch_lf(monkeypatch)
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    client._anthropic.messages.create.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await client.chat(prompt="q", model="haiku")

    assert client._anthropic.messages.create.call_count == 3
    lf_calls["end_err"].assert_called_once()
    err_kwargs = lf_calls["end_err"].call_args
    # end_err signature: (gen, error, extra_metadata=...)
    expected = {"retries_exhausted": 3}
    assert (
        err_kwargs.args[2] == expected
        or err_kwargs.kwargs.get("extra_metadata") == expected
    )


@pytest.mark.asyncio
async def test_chat_anthropic_empty_content_retries(client: LLMClient, monkeypatch):
    _patch_lf(monkeypatch)
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    empty = SimpleNamespace(content=[], usage=SimpleNamespace(input_tokens=0, output_tokens=0))
    client._anthropic.messages.create.side_effect = [empty, _fake_anthropic_msg(text="ok")]

    resp = await client.chat(prompt="q", model="haiku")
    assert resp.text == "ok"
    assert resp.attempt == 2


# ---------------------------------------------------------- DeepSeek
@pytest.mark.asyncio
async def test_chat_deepseek_happy(monkeypatch):
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value={
        "choices": [{"message": {"content": "ds reply"}}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 7},
    })
    mock_http.post.return_value = fake_resp

    client = LLMClient(deepseek_api_key="ds-k", http_client=mock_http, backoff_base=1)
    _patch_lf(monkeypatch)

    resp = await client.chat(prompt="q", model="deepseek", name="ds.test")
    assert resp.text == "ds reply"
    assert resp.model == "deepseek-chat"
    assert resp.usage.input == 15
    assert resp.usage.output == 7
    # cost: (15*0.27 + 7*1.10) / 1e6
    assert resp.cost_usd == pytest.approx((15 * 0.27 + 7 * 1.10) / 1_000_000)

    # no auth header leaked to prompt
    call = mock_http.post.call_args
    assert call.kwargs["headers"]["Authorization"] == "Bearer ds-k"
    assert call.kwargs["json"]["model"] == "deepseek-chat"


@pytest.mark.asyncio
async def test_chat_deepseek_no_key_raises():
    client = LLMClient()  # no keys
    with pytest.raises(RuntimeError, match="deepseek_api_key"):
        await client.chat(prompt="q", model="deepseek-chat")


@pytest.mark.asyncio
async def test_chat_anthropic_no_key_raises():
    client = LLMClient()  # no keys
    with pytest.raises(RuntimeError, match="anthropic_api_key"):
        await client.chat(prompt="q", model="haiku")


# ---------------------------------------------------------- structured
@pytest.mark.asyncio
async def test_chat_structured_happy(client: LLMClient, monkeypatch):
    from pydantic import BaseModel

    class MyModel(BaseModel):
        answer: str
        score: int

    _patch_lf(monkeypatch)

    instructor_mock = MagicMock()
    pydantic_instance = MyModel(answer="42", score=99)
    raw_resp = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=200, output_tokens=30,
            cache_creation_input_tokens=0, cache_read_input_tokens=200,
        ),
    )
    instructor_mock.messages = MagicMock()
    instructor_mock.messages.create_with_completion = AsyncMock(
        return_value=(pydantic_instance, raw_resp),
    )
    client._instructor_client = instructor_mock

    obj, resp = await client.chat_structured(
        prompt="q", response_model=MyModel, model="sonnet",
        system="system-prompt", system_cache=True, name="agent.analyze",
    )

    assert obj is pydantic_instance
    assert obj.answer == "42"
    assert resp.usage.input == 200
    assert resp.usage.cache_read == 200
    assert resp.model == "claude-sonnet-4-20250514"

    call_kwargs = instructor_mock.messages.create_with_completion.call_args.kwargs
    assert call_kwargs["response_model"] is MyModel
    assert isinstance(call_kwargs["system"], list)
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_chat_structured_no_instructor(monkeypatch):
    """When instructor package is not available, chat_structured raises RuntimeError."""
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    client = LLMClient(anthropic_api_key="sk-test")
    monkeypatch.setattr("agents_core.llm.client._INSTRUCTOR_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="instructor"):
        await client.chat_structured(prompt="q", response_model=M)


# ---------------------------------------------------------- aclose semantics
@pytest.mark.asyncio
async def test_owns_http_closes_on_aclose():
    c = LLMClient(anthropic_api_key="sk")
    assert c._owns_http is True
    await c.aclose()


@pytest.mark.asyncio
async def test_passed_http_not_owned():
    shared = httpx.AsyncClient()
    c = LLMClient(anthropic_api_key="sk", http_client=shared)
    assert c._owns_http is False
    await c.aclose()
    assert shared.is_closed is False
    await shared.aclose()


@pytest.mark.asyncio
async def test_async_context_manager():
    c = LLMClient(anthropic_api_key="sk")
    async with c:
        pass  # aenter/aexit covered


# ---------------------------------------------------------- DeepSeek retries
@pytest.mark.asyncio
async def test_deepseek_retry_then_success(monkeypatch):
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    fake_ok = MagicMock()
    fake_ok.raise_for_status = MagicMock()
    fake_ok.json = MagicMock(return_value={
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    })
    mock_http.post.side_effect = [httpx.ConnectError("boom"), fake_ok]

    client = LLMClient(deepseek_api_key="k", http_client=mock_http, backoff_base=1)
    _patch_lf(monkeypatch)
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    resp = await client.chat(prompt="q", model="deepseek-chat")
    assert resp.text == "ok"
    assert resp.attempt == 2
    assert mock_http.post.call_count == 2


@pytest.mark.asyncio
async def test_deepseek_retries_exhausted(monkeypatch):
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.post.side_effect = httpx.ConnectError("dead")

    client = LLMClient(deepseek_api_key="k", http_client=mock_http, backoff_base=1)
    lf_calls = _patch_lf(monkeypatch)
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    with pytest.raises(httpx.ConnectError):
        await client.chat(prompt="q", model="deepseek-chat")
    assert mock_http.post.call_count == 3
    lf_calls["end_err"].assert_called_once()


@pytest.mark.asyncio
async def test_deepseek_malformed_response(monkeypatch):
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    fake = MagicMock()
    fake.raise_for_status = MagicMock()
    fake.json = MagicMock(return_value={"not": "choices"})
    mock_http.post.return_value = fake

    client = LLMClient(deepseek_api_key="k", http_client=mock_http, backoff_base=1)
    _patch_lf(monkeypatch)
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    with pytest.raises(RuntimeError, match="malformed"):
        await client.chat(prompt="q", model="deepseek-chat")


# ---------------------------------------------------------- structured retry
@pytest.mark.asyncio
async def test_chat_structured_retries_then_success(client: LLMClient, monkeypatch):
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    _patch_lf(monkeypatch)
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    instructor_mock = MagicMock()
    good_raw = SimpleNamespace(usage=SimpleNamespace(
        input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0,
    ))
    instructor_mock.messages = MagicMock()
    instructor_mock.messages.create_with_completion = AsyncMock(
        side_effect=[RuntimeError("validation flaky"), (M(x=42), good_raw)]
    )
    client._instructor_client = instructor_mock

    obj, resp = await client.chat_structured(prompt="q", response_model=M)
    assert obj.x == 42
    assert resp.attempt == 2
    assert instructor_mock.messages.create_with_completion.call_count == 2


@pytest.mark.asyncio
async def test_chat_structured_no_anthropic_key(monkeypatch):
    """chat_structured requires Anthropic key even when instructor available."""
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    client = LLMClient()  # no keys
    with pytest.raises(RuntimeError, match="anthropic"):
        await client.chat_structured(prompt="q", response_model=M)


@pytest.mark.asyncio
async def test_model_shortcut_unknown_passes_through(client: LLMClient, monkeypatch):
    """Unknown model string is passed through verbatim (allows new model ids)."""
    _patch_lf(monkeypatch)
    client._anthropic.messages.create.return_value = _fake_anthropic_msg()
    resp = await client.chat(prompt="q", model="claude-some-future-id-9000")
    assert resp.model == "claude-some-future-id-9000"
