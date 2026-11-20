"""Tests for the local LLM client in codesearch/generation/llm.py: OllamaLLM.

The project runs fully on local models through Ollama (free, no API key).
No network here: OllamaLLM gets an httpx.MockTransport (real requests, fake server).
One test talks to a real local Ollama (marked needs_llm; needs `ollama serve` and
the model pulled). Streaming tests use a fake NDJSON body that records being closed.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import aclosing
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
# OllamaLLM.stream (Phase 3): NDJSON tokens, errors, and closing upstream on cancel
# =====================================================================


class LineStream(httpx.AsyncByteStream):
    """A streaming response body, one NDJSON line at a time. Records whether it was closed.

    With ``hang=True`` it never ends after its lines: like a model still generating.
    """

    def __init__(self, lines: list[Any], hang: bool = False) -> None:
        self.lines = lines
        self.hang = hang
        self.sent = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for line in self.lines:
            self.sent += 1
            text = line if isinstance(line, str) else json.dumps(line)
            yield (text + "\n").encode()
        if self.hang:
            await asyncio.Event().wait()  # never set: the "model" keeps going

    async def aclose(self) -> None:
        self.closed = True


class StreamServer:
    """Fake Ollama for streaming: answers every request with the given body or status."""

    def __init__(
        self, body: LineStream | None = None, status: int = 200, raise_exc: Exception | None = None
    ) -> None:
        self.body = body or LineStream([])
        self.status = status
        self.raise_exc = raise_exc
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raise_exc is not None:
            raise self.raise_exc
        return httpx.Response(self.status, stream=self.body)

    def llm(self, **kwargs: Any) -> OllamaLLM:
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        return OllamaLLM(model="gemma3:4b", client=client, **kwargs)


def token(text: str) -> dict[str, Any]:
    return {"model": "gemma3:4b", "response": text, "done": False}


DONE = {"model": "gemma3:4b", "response": "", "done": True, "eval_count": 3}


async def collect(llm: OllamaLLM, prompt: str = "q", system: str | None = None) -> list[str]:
    async with aclosing(llm.stream(prompt, system=system)) as tokens:
        return [t async for t in tokens]


async def test_stream_yields_tokens_in_order_and_stops_at_done() -> None:
    server = StreamServer(LineStream([token("Use"), token(" useState"), token(" [1]."), DONE]))
    assert await collect(server.llm()) == ["Use", " useState", " [1]."]


async def test_stream_posts_a_streaming_request() -> None:
    server = StreamServer(LineStream([token("a"), DONE]))
    await collect(server.llm(temperature=0.2, max_tokens=50), prompt="P", system="S")
    request = server.requests[0]
    assert request.method == "POST"
    assert request.url == "http://localhost:11434/api/generate"
    body = json.loads(request.content)
    assert body["stream"] is True
    assert (body["prompt"], body["system"], body["model"]) == ("P", "S", "gemma3:4b")
    assert body["options"] == {"temperature": 0.2, "num_predict": 50}


async def test_stream_leaves_out_an_empty_system_prompt() -> None:
    server = StreamServer(LineStream([DONE]))
    await collect(server.llm(), system=None)
    assert "system" not in json.loads(server.requests[0].content)


async def test_stream_skips_blank_lines_and_empty_tokens() -> None:
    server = StreamServer(LineStream(["", token(""), token("x"), "   ", DONE]))
    assert await collect(server.llm()) == ["x"]


async def test_stream_ignores_anything_after_done() -> None:
    server = StreamServer(LineStream([token("a"), DONE, token("late")]))
    assert await collect(server.llm()) == ["a"]


async def test_stream_error_line_is_unavailable() -> None:
    server = StreamServer(LineStream([token("a"), {"error": "model 'gemma3:4b' not found"}]))
    with pytest.raises(LLMUnavailableError, match="not found"):
        await collect(server.llm())


async def test_stream_malformed_line_is_unavailable() -> None:
    server = StreamServer(LineStream(["{not json"]))
    with pytest.raises(LLMUnavailableError):
        await collect(server.llm())


@pytest.mark.parametrize("status", [404, 500, 503])
async def test_stream_http_error_status_is_unavailable(status: int) -> None:
    server = StreamServer(status=status)
    with pytest.raises(LLMUnavailableError):
        await collect(server.llm())


async def test_stream_connection_failure_is_unavailable() -> None:
    server = StreamServer(raise_exc=httpx.ConnectError("refused"))
    with pytest.raises(LLMUnavailableError):
        await collect(server.llm())


async def test_stream_timeout_is_a_timeout() -> None:
    server = StreamServer(raise_exc=httpx.ReadTimeout("slow"))
    with pytest.raises(LLMTimeoutError):
        await collect(server.llm())


async def test_stream_closes_upstream_when_the_consumer_stops() -> None:
    """The user closed the tab after one token: Ollama's connection must close too."""
    body = LineStream([token("first"), token("second")], hang=True)
    server = StreamServer(body)
    async with aclosing(server.llm().stream("q")) as tokens:
        async for first in tokens:
            assert first == "first"
            break
    assert body.closed


async def test_stream_closes_upstream_when_the_task_is_cancelled() -> None:
    """Cancelled while waiting for the next token (the model is still generating)."""
    body = LineStream([token("first")], hang=True)
    server = StreamServer(body)
    got: list[str] = []

    async def consume() -> None:
        async with aclosing(server.llm().stream("q")) as tokens:
            async for t in tokens:
                got.append(t)

    task = asyncio.create_task(consume())
    for _ in range(20):
        await asyncio.sleep(0)
    assert got == ["first"]  # now waiting on a model that never finishes
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert body.closed


async def test_generate_still_works_after_the_refactor() -> None:
    """generate() and stream() should build the same request body (stream flag aside)."""
    server = OllamaServer()
    llm = ollama(server)
    assert await llm.generate("P", system="S") == "Local answer [1]."
    assert json.loads(server.requests[0].content)["stream"] is False
