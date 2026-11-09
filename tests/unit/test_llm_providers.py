"""Tests for the local LLM client in codesearch/generation/llm.py: OllamaLLM.

The project runs fully on local models through Ollama (free, no API key).
No network here: OllamaLLM gets an httpx.MockTransport (real requests, fake server).
One test talks to a real local Ollama (marked needs_llm; needs `ollama serve` and
the model pulled). Streaming (stream()) comes in Phase 3: skipped section at the end.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from codesearch.config import Settings
from codesearch.errors import LLMTimeoutError, LLMUnavailableError
from codesearch.factory import build_llm
from codesearch.generation.llm import FakeLLM, OllamaLLM

# =====================================================================
# OllamaLLM.generate (httpx.MockTransport: real requests, fake server)
# =====================================================================


class OllamaServer:
    def __init__(self, reply: Any = None, status: int = 200, raise_exc: Exception | None = None):
        self.reply = reply if reply is not None else {"response": "Local answer [1]."}
        self.status = status
        self.raise_exc = raise_exc
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raise_exc is not None:
            raise self.raise_exc
        return httpx.Response(self.status, json=self.reply)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def ollama(server: OllamaServer, **kwargs: Any) -> OllamaLLM:
    options: dict[str, Any] = {"model": "gemma3:4b", "base_url": "http://localhost:11434"}
    options.update(kwargs)
    return OllamaLLM(client=server.client(), **options)


async def test_ollama_returns_the_response_text() -> None:
    assert await ollama(OllamaServer()).generate("Q?") == "Local answer [1]."


async def test_ollama_posts_to_the_generate_endpoint() -> None:
    server = OllamaServer()
    await ollama(server).generate("Q?")
    (request,) = server.requests
    assert request.method == "POST"
    assert str(request.url) == "http://localhost:11434/api/generate"


async def test_ollama_trailing_slash_in_base_url_is_handled() -> None:
    server = OllamaServer()
    await ollama(server, base_url="http://localhost:11434/").generate("Q?")
    assert str(server.requests[0].url) == "http://localhost:11434/api/generate"


async def test_ollama_request_body() -> None:
    server = OllamaServer()
    llm = ollama(server, temperature=0.1, max_tokens=256)
    await llm.generate("the prompt", system="be brief")
    body = json.loads(server.requests[0].content)
    assert body["model"] == "gemma3:4b"
    assert body["prompt"] == "the prompt"
    assert body["system"] == "be brief"
    assert body["stream"] is False  # one JSON reply, not a stream
    assert body["options"] == {"temperature": 0.1, "num_predict": 256}  # max_tokens -> num_predict


async def test_ollama_leaves_out_an_empty_system_prompt() -> None:
    server = OllamaServer()
    await ollama(server).generate("the prompt")
    assert "system" not in json.loads(server.requests[0].content)


async def test_ollama_connection_failure_is_unavailable() -> None:
    server = OllamaServer(raise_exc=httpx.ConnectError("Connection refused"))
    with pytest.raises(LLMUnavailableError):
        await ollama(server).generate("Q?")


async def test_ollama_timeout_is_a_timeout() -> None:
    server = OllamaServer(raise_exc=httpx.ReadTimeout("slow"))
    with pytest.raises(LLMTimeoutError):
        await ollama(server).generate("Q?")


@pytest.mark.parametrize("status", [404, 500])
async def test_ollama_http_error_status_is_unavailable(status: int) -> None:
    """404 usually means the model isn't pulled (ollama pull gemma3:4b)."""
    server = OllamaServer(reply={"error": "model not found"}, status=status)
    with pytest.raises(LLMUnavailableError):
        await ollama(server).generate("Q?")


async def test_ollama_reply_without_response_field_is_unavailable() -> None:
    with pytest.raises(LLMUnavailableError):
        await ollama(OllamaServer(reply={"oops": 1})).generate("Q?")


def test_ollama_defaults() -> None:
    llm = OllamaLLM()
    assert (llm.model, llm.base_url) == ("gemma3:4b", "http://localhost:11434")


# =====================================================================
# FakeLLM accepts a system prompt too (same interface for every LLM)
# =====================================================================


async def test_fake_llm_accepts_a_system_prompt() -> None:
    fake = FakeLLM(responses=["canned"])
    assert await fake.generate("Q?", system="rules") == "canned"
    assert fake.prompts == ["Q?"]
    assert fake.systems == ["rules"]


async def test_fake_llm_records_none_when_no_system_prompt() -> None:
    fake = FakeLLM()
    await fake.generate("Q?")
    assert fake.systems == [None]


# =====================================================================
# factory.build_llm
# =====================================================================


def settings(**llm: Any) -> Settings:
    return Settings(_env_file=None).with_overrides(llm=llm)  # type: ignore[call-arg]


def test_factory_builds_fake_by_default() -> None:
    assert isinstance(build_llm(settings()), FakeLLM)


def test_llm_defaults_are_local() -> None:
    llm = settings().llm
    assert (llm.provider, llm.model) == ("fake", "gemma3:4b")  # fake in tests, Ollama model name
    assert llm.ollama_base_url == "http://localhost:11434"


def test_factory_rejects_groq_now_that_the_project_is_local_only() -> None:
    with pytest.raises(ValidationError):
        settings(provider="groq")


def test_factory_builds_ollama_from_settings() -> None:
    llm = build_llm(
        settings(provider="ollama", model="gemma3:4b", ollama_base_url="http://box:11434")
    )
    assert isinstance(llm, OllamaLLM)
    assert (llm.model, llm.base_url) == ("gemma3:4b", "http://box:11434")


# =====================================================================
# A real local Ollama (ollama serve; ollama pull gemma3:4b; pytest -m needs_llm)
# =====================================================================


@pytest.mark.needs_llm
async def test_real_ollama_answers() -> None:
    model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
    llm = OllamaLLM(model=model, max_tokens=20)
    try:
        text = await llm.generate("Reply with the single word: pong")
    except LLMUnavailableError as error:
        pytest.skip(f"no local Ollama with {model}: {error}")
    assert "pong" in text.lower()


# =====================================================================
# Phase 3: streaming (write stream() then remove the skip)
# =====================================================================


@pytest.mark.skip(reason="Phase 3: OllamaLLM.stream with cancellation")
async def test_ollama_stream_closes_upstream_when_the_consumer_stops() -> None:
    raise AssertionError
