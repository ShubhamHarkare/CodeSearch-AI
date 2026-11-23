"""OpenAICompatLLM: a hosted model behind the OpenAI chat-completions API (Groq).

No network: httpx.MockTransport plays the server. Streaming uses a fake SSE body
that records being closed. One needs_llm test calls the real API with the
provider, model and key from .env.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from codesearch.config import Settings
from codesearch.errors import (
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from codesearch.factory import build_llm
from codesearch.generation.llm import OpenAICompatLLM

KEY = "gsk_test_secret_123"
BASE = "https://api.groq.com/openai/v1"
MODEL = "llama-3.1-8b-instant"


def completion(text: str | None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


class Server:
    """Fake chat-completions server: one fixed reply (or exception) for every request."""

    def __init__(
        self,
        reply: Any = None,
        status: int = 200,
        headers: dict[str, str] | None = None,
        raise_exc: Exception | None = None,
        stream: httpx.AsyncByteStream | None = None,
    ) -> None:
        self.reply = reply if reply is not None else completion("Hosted answer [1].")
        self.status = status
        self.headers = headers or {}
        self.raise_exc = raise_exc
        self.stream = stream
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.stream is not None:
            return httpx.Response(self.status, headers=self.headers, stream=self.stream)
        return httpx.Response(self.status, headers=self.headers, json=self.reply)

    def llm(self, **kwargs: Any) -> OpenAICompatLLM:
        options: dict[str, Any] = {"model": MODEL, "api_key": KEY, "base_url": BASE}
        options.update(kwargs)
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        return OpenAICompatLLM(client=client, **options)

    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)  # type: ignore[no-any-return]


# =====================================================================
# construction
# =====================================================================


def test_an_empty_api_key_is_refused_at_startup() -> None:
    # Fail when the app starts, not on the first user's question.
    with pytest.raises(ValueError):
        OpenAICompatLLM(MODEL, api_key="")


def test_the_key_is_not_in_the_repr() -> None:
    # reprs end up in logs and tracebacks
    assert KEY not in repr(Server().llm())


# =====================================================================
# generate
# =====================================================================


async def test_generate_returns_the_message_content() -> None:
    assert await Server().llm().generate("Q?") == "Hosted answer [1]."


async def test_generate_posts_to_chat_completions_with_a_bearer_token() -> None:
    server = Server()
    await server.llm().generate("Q?")
    (request,) = server.requests
    assert request.method == "POST"
    assert str(request.url) == f"{BASE}/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {KEY}"


async def test_a_trailing_slash_in_the_base_url_is_handled() -> None:
    server = Server()
    await server.llm(base_url=BASE + "/").generate("Q?")
    assert str(server.requests[0].url) == f"{BASE}/chat/completions"


async def test_the_request_body_uses_chat_messages() -> None:
    server = Server()
    await server.llm(temperature=0.1, max_tokens=256).generate("the prompt", system="be brief")
    body = server.body()
    assert body["model"] == MODEL
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},  # the rules go first, as the system message
        {"role": "user", "content": "the prompt"},
    ]
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 256
    assert body["stream"] is False


async def test_no_system_message_without_a_system_prompt() -> None:
    server = Server()
    await server.llm().generate("the prompt")
    assert server.body()["messages"] == [{"role": "user", "content": "the prompt"}]


async def test_a_null_content_is_an_empty_answer() -> None:
    assert await Server(reply=completion(None)).llm().generate("Q?") == ""


@pytest.mark.parametrize(
    "reply", [{"oops": 1}, {"choices": []}, {"choices": [{"message": {}}]}, ["not", "a", "dict"]]
)
async def test_an_unexpected_reply_shape_is_unavailable(reply: Any) -> None:
    with pytest.raises(LLMUnavailableError):
        await Server(reply=reply).llm().generate("Q?")


async def test_429_is_rate_limited_and_mentions_the_wait() -> None:
    server = Server(
        reply={"error": {"message": "Rate limit reached", "type": "tokens"}},
        status=429,
        headers={"retry-after": "7"},
    )
    with pytest.raises(LLMRateLimitedError, match="7"):
        await server.llm().generate("Q?")


def test_rate_limited_is_a_429_for_our_clients_and_still_an_unavailable_error() -> None:
    # The API answers our users with 429 "busy, try again" (not 503 "down"), and any
    # code that already handles LLMUnavailableError keeps working.
    assert LLMRateLimitedError.http_status == 429
    assert issubclass(LLMRateLimitedError, LLMUnavailableError)


@pytest.mark.parametrize("status", [401, 403, 404, 500, 503])
async def test_an_error_status_is_unavailable(status: int) -> None:
    server = Server(reply={"error": {"message": "nope"}}, status=status)
    with pytest.raises(LLMUnavailableError) as info:
        await server.llm().generate("Q?")
    assert not isinstance(info.value, LLMRateLimitedError)


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_errors_never_contain_the_api_key(status: int) -> None:
    # Error text is logged (main.py logs str(exc)): a leaked key would sit in the logs.
    server = Server(reply={"error": {"message": f"bad key {KEY}"}}, status=status)
    with pytest.raises(LLMUnavailableError) as info:
        await server.llm().generate("Q?")
    assert KEY not in str(info.value)
    assert KEY not in repr(info.value)


async def test_a_connection_failure_is_unavailable() -> None:
    server = Server(raise_exc=httpx.ConnectError("no route to host"))
    with pytest.raises(LLMUnavailableError):
        await server.llm().generate("Q?")


async def test_a_timeout_is_a_timeout() -> None:
    server = Server(raise_exc=httpx.ReadTimeout("slow"))
    with pytest.raises(LLMTimeoutError):
        await server.llm().generate("Q?")


# =====================================================================
# stream: server-sent events, "data: {...}" lines, ending with "data: [DONE]"
# =====================================================================


class SSEStream(httpx.AsyncByteStream):
    """A streaming body sent line by line. Records whether it was closed.

    With ``hang=True`` it never ends after its lines: like a model still generating.
    """

    def __init__(self, lines: list[str], hang: bool = False) -> None:
        self.lines = lines
        self.hang = hang
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for line in self.lines:
            yield (line + "\n").encode()
        if self.hang:
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


def chunk(text: str | None, finish: str | None = None) -> str:
    delta = {} if text is None else {"content": text}
    data = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return "data: " + json.dumps(data)


ROLE = "data: " + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
DONE = "data: [DONE]"


async def collect(llm: OpenAICompatLLM, prompt: str = "q", system: str | None = None) -> list[str]:
    async with aclosing(llm.stream(prompt, system=system)) as tokens:
        return [t async for t in tokens]


async def test_stream_yields_tokens_in_order_and_stops_at_done() -> None:
    body = SSEStream(
        [ROLE, "", chunk("Use "), "", chunk("useState"), "", chunk(None, "stop"), DONE]
    )
    assert await collect(Server(stream=body).llm()) == ["Use ", "useState"]


async def test_stream_sends_a_streaming_chat_request() -> None:
    server = Server(stream=SSEStream([chunk("hi"), DONE]))
    await collect(server.llm(), "the prompt", system="be brief")
    body = server.body()
    assert body["stream"] is True
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert server.requests[0].headers["Authorization"] == f"Bearer {KEY}"


async def test_stream_skips_comments_blank_lines_and_empty_deltas() -> None:
    # ": keep-alive" is an SSE comment; some servers send them while the model thinks.
    lines = [": keep-alive", "", ROLE, chunk(""), chunk("a"), "event: ping", chunk(None), DONE]
    assert await collect(Server(stream=SSEStream(lines)).llm()) == ["a"]


async def test_stream_ignores_anything_after_done() -> None:
    body = SSEStream([chunk("a"), DONE, chunk("late")])
    assert await collect(Server(stream=body).llm()) == ["a"]


async def test_stream_error_event_is_unavailable() -> None:
    body = SSEStream([chunk("a"), "data: " + json.dumps({"error": {"message": "overloaded"}})])
    with pytest.raises(LLMUnavailableError):
        await collect(Server(stream=body).llm())


async def test_stream_malformed_line_is_unavailable() -> None:
    with pytest.raises(LLMUnavailableError):
        await collect(Server(stream=SSEStream(["data: {not json"])).llm())


async def test_stream_429_is_rate_limited() -> None:
    server = Server(
        status=429, headers={"retry-after": "3"}, stream=SSEStream(['{"error": "slow down"}'])
    )
    with pytest.raises(LLMRateLimitedError):
        await collect(server.llm())


@pytest.mark.parametrize("status", [401, 500])
async def test_stream_error_status_is_unavailable(status: int) -> None:
    server = Server(status=status, stream=SSEStream(['{"error": {"message": "nope"}}']))
    with pytest.raises(LLMUnavailableError):
        await collect(server.llm())


async def test_stream_connection_failure_is_unavailable() -> None:
    with pytest.raises(LLMUnavailableError):
        await collect(Server(raise_exc=httpx.ConnectError("down")).llm())


async def test_stream_timeout_is_a_timeout() -> None:
    with pytest.raises(LLMTimeoutError):
        await collect(Server(raise_exc=httpx.ReadTimeout("slow")).llm())


async def test_stream_closes_upstream_when_the_consumer_stops() -> None:
    # The user closed the tab: stop paying for tokens nobody will read.
    body = SSEStream([chunk("first"), chunk("second")], hang=True)
    async with aclosing(Server(stream=body).llm().stream("q")) as tokens:
        async for first in tokens:
            assert first == "first"
            break
    assert body.closed


async def test_stream_closes_upstream_when_the_task_is_cancelled() -> None:
    body = SSEStream([chunk("first")], hang=True)
    llm = Server(stream=body).llm()
    got: list[str] = []

    async def consume() -> None:
        async with aclosing(llm.stream("q")) as tokens:
            async for t in tokens:
                got.append(t)

    task = asyncio.create_task(consume())
    for _ in range(20):
        await asyncio.sleep(0)
    assert got == ["first"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert body.closed


# =====================================================================
# settings + factory
# =====================================================================


def settings(**llm: Any) -> Settings:
    return Settings(_env_file=None).with_overrides(llm=llm)  # type: ignore[call-arg]


def test_the_default_base_url_is_groq() -> None:
    assert settings().llm.openai_base_url == BASE


def test_the_api_key_is_secret_in_settings() -> None:
    s = settings(provider="openai", api_key=KEY)
    assert KEY not in repr(s)  # SecretStr prints as '**********'
    assert s.llm.api_key.get_secret_value() == KEY


def test_the_factory_builds_the_hosted_llm() -> None:
    llm = build_llm(settings(provider="openai", model=MODEL, api_key=KEY, max_tokens=300))
    assert isinstance(llm, OpenAICompatLLM)
    assert (llm.model, llm.base_url, llm.max_tokens) == (MODEL, BASE, 300)


def test_the_factory_refuses_the_hosted_llm_without_a_key() -> None:
    with pytest.raises(ValueError):
        build_llm(settings(provider="openai", model=MODEL))


def test_the_key_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODESEARCH_LLM__PROVIDER", "openai")
    monkeypatch.setenv("CODESEARCH_LLM__API_KEY", KEY)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm.provider == "openai"
    assert s.llm.api_key.get_secret_value() == KEY


# =====================================================================
# The real API (uses the key and model from .env; pytest -m needs_llm)
# =====================================================================


@pytest.mark.needs_llm
async def test_real_hosted_llm_answers_and_streams() -> None:
    # Settings() reads .env exactly as the app does: no need to export anything.
    llm_settings = Settings().llm
    if llm_settings.provider != "openai" or not llm_settings.api_key.get_secret_value():
        pytest.skip("set CODESEARCH_LLM__PROVIDER=openai and CODESEARCH_LLM__API_KEY in .env")
    llm = build_llm(Settings())  # the same object the app builds from .env
    assert isinstance(llm, OpenAICompatLLM)
    assert "pong" in (await llm.generate("Reply with the single word: pong")).lower()
    assert "pong" in "".join(await collect(llm, "Reply with the single word: pong")).lower()


# =====================================================================
# Reasoning models (gpt-oss): keep the hidden thinking short
# =====================================================================


async def test_reasoning_effort_is_sent_when_set() -> None:
    # gpt-oss "thinks" before answering, and the thinking counts toward max_tokens.
    server = Server()
    await server.llm(reasoning_effort="low").generate("Q?")
    assert server.body()["reasoning_effort"] == "low"


async def test_reasoning_effort_is_left_out_by_default() -> None:
    # Models that don't reason reject the field, so only send it when asked.
    server = Server()
    await server.llm().generate("Q?")
    assert "reasoning_effort" not in server.body()


async def test_streamed_reasoning_is_not_part_of_the_answer() -> None:
    # Groq streams the thinking in delta.reasoning and the answer in delta.content.
    thinking = "data: " + json.dumps({"choices": [{"delta": {"reasoning": "The user asks..."}}]})
    body = SSEStream([thinking, chunk("Use "), chunk("useState"), DONE])
    assert await collect(Server(stream=body).llm()) == ["Use ", "useState"]


def test_the_factory_passes_reasoning_effort() -> None:
    llm = build_llm(
        settings(provider="openai", model="openai/gpt-oss-20b", api_key=KEY, reasoning_effort="low")
    )
    assert isinstance(llm, OpenAICompatLLM)
    assert llm.reasoning_effort == "low"


def test_reasoning_effort_must_be_a_known_level() -> None:
    with pytest.raises(ValidationError):
        settings(reasoning_effort="maximum")
