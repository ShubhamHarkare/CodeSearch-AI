"""Tests for codesearch/ui/api_client.py: the UI's only way to reach the API.

No server needed: httpx.MockTransport plays the API, so each test controls exactly
what comes back (an answer, a 503, garbage, a dead connection).
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from codesearch.ui.api_client import ApiClient, ApiError

BASE = "http://api.test"
ANSWER = {
    "answer": "Use an updater function [1].",
    "citations": [{"number": 1, "chunk_id": "c1", "url": "https://react.dev/x", "title": "X"}],
    "abstained": False,
    "confidence": None,
    "groundedness": None,
    "cache_tier": None,
    "timings": {"total_ms": 12.0},
    "trace_id": "t-1",
    "index_version": "iv",
    "prompt_version": "1",
}

Handler = Callable[[httpx.Request], httpx.Response]


def make(handler: Handler, **kwargs: Any) -> ApiClient:
    return ApiClient(BASE, transport=httpx.MockTransport(handler), **kwargs)


class Recorder:
    """A fake API that records each request and answers with a fixed response."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response

    @property
    def last_json(self) -> Any:
        return json.loads(self.requests[-1].content)


# =====================================================================
# ask
# =====================================================================


def test_ask_posts_the_question_and_returns_the_body() -> None:
    api = Recorder(httpx.Response(200, json=ANSWER))
    result = make(api).ask("How do I update state?")
    request = api.requests[0]
    assert request.method == "POST"
    assert request.url == f"{BASE}/v1/ask"
    assert api.last_json == {
        "question": "How do I update state?",
        "history": [],
        "bypass_cache": False,
    }
    assert result == ANSWER


def test_ask_sends_history_and_bypass_cache() -> None:
    api = Recorder(httpx.Response(200, json=ANSWER))
    history = [
        {"role": "user", "content": "What is useState?"},
        {"role": "assistant", "content": "A Hook [1]."},
    ]
    make(api).ask("and objects?", history, bypass_cache=True)
    assert api.last_json["history"] == history
    assert api.last_json["bypass_cache"] is True


def test_a_trailing_slash_in_the_base_url_is_fine() -> None:
    api = Recorder(httpx.Response(200, json=ANSWER))
    ApiClient(BASE + "/", transport=httpx.MockTransport(api)).ask("q")
    assert api.requests[0].url == f"{BASE}/v1/ask"


def test_an_abstained_answer_is_a_normal_result() -> None:
    abstained = {**ANSWER, "abstained": True, "citations": []}
    assert (
        make(Recorder(httpx.Response(200, json=abstained))).ask("capital of France?") == abstained
    )


# =====================================================================
# Errors become ApiError with a message fit to show in the UI
# =====================================================================


def test_a_503_carries_the_apis_public_message_and_trace_id() -> None:
    body = {"error": "The language model is unavailable.", "trace_id": "t-9"}
    with pytest.raises(ApiError) as caught:
        make(Recorder(httpx.Response(503, json=body))).ask("q")
    assert caught.value.status == 503
    assert caught.value.trace_id == "t-9"
    assert str(caught.value) == "The language model is unavailable."


def test_a_500_is_an_api_error() -> None:
    body = {"error": "An internal server error occurred.", "trace_id": "t-5"}
    with pytest.raises(ApiError) as caught:
        make(Recorder(httpx.Response(500, json=body))).ask("q")
    assert caught.value.status == 500


def test_a_422_says_what_was_wrong_with_the_request() -> None:
    body = {
        "detail": [{"loc": ["body", "question"], "msg": "String should have at least 1 character"}]
    }
    with pytest.raises(ApiError) as caught:
        make(Recorder(httpx.Response(422, json=body))).ask("")
    assert caught.value.status == 422
    assert str(caught.value).startswith("Invalid request")
    assert "at least 1 character" in str(caught.value)


def test_an_error_that_is_not_json_still_gives_a_message() -> None:
    """A proxy in front of the API can answer with an HTML 502 page."""
    with pytest.raises(ApiError) as caught:
        make(Recorder(httpx.Response(502, text="<html>Bad Gateway</html>"))).ask("q")
    assert caught.value.status == 502
    assert "502" in str(caught.value)
    assert caught.value.trace_id is None


def test_api_down_gives_a_helpful_message() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ApiError) as caught:
        make(refuse).ask("q")
    assert caught.value.status is None
    assert BASE in str(caught.value)  # tells the user which address it tried


def test_a_timeout_gives_a_helpful_message() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(ApiError) as caught:
        make(slow, timeout_s=5).ask("q")
    assert caught.value.status is None
    assert "timed out" in str(caught.value).lower()


def test_api_error_is_a_plain_exception_the_ui_can_catch() -> None:
    error = ApiError("boom", status=503, trace_id="t")
    assert isinstance(error, Exception)
    assert (str(error), error.status, error.trace_id) == ("boom", 503, "t")
    assert ApiError("x").status is None and ApiError("x").trace_id is None


# =====================================================================
# ready / index_info (the sidebar)
# =====================================================================


def test_ready_returns_the_status_body() -> None:
    body = {"status": "ok", "components": {"index": "ok", "cache": "ok"}, "versions": {}}
    api = Recorder(httpx.Response(200, json=body))
    assert make(api).ready() == body
    assert api.requests[0].method == "GET"
    assert api.requests[0].url == f"{BASE}/readyz"


def test_ready_returns_the_body_on_503_too() -> None:
    """503 from /readyz is an answer ("not ready yet"), not a failure to show as an error."""
    body = {"status": "unavailable", "components": {"index": "missing"}, "versions": {}}
    assert make(Recorder(httpx.Response(503, json=body))).ready() == body


def test_ready_raises_when_the_api_is_unreachable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ApiError):
        make(refuse).ready()


def test_index_info() -> None:
    body = {"index_version": "iv", "corpus_sha": "abc", "chunk_count": 10}
    api = Recorder(httpx.Response(200, json=body))
    assert make(api).index_info() == body
    assert api.requests[0].url == f"{BASE}/v1/index"


def test_index_info_before_the_index_loads_raises() -> None:
    body = {"error": "The search index is still loading.", "trace_id": "t"}
    with pytest.raises(ApiError) as caught:
        make(Recorder(httpx.Response(503, json=body))).index_info()
    assert caught.value.status == 503


# =====================================================================
# Lifecycle and boundaries
# =====================================================================


def test_the_client_is_reused_and_can_be_closed() -> None:
    api = Recorder(httpx.Response(200, json=ANSWER))
    with make(api) as client:
        client.ask("a")
        client.ask("b")
    assert len(api.requests) == 2


def test_api_client_only_imports_schemas_from_codesearch() -> None:
    """The UI must stay a pure HTTP client: no pipeline, models or Redis."""
    import codesearch.ui.api_client as module

    tree = ast.parse(Path(module.__file__).read_text())
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    ours = {name for name in imported if name.split(".")[0] == "codesearch"}
    assert ours <= {"codesearch.schemas"}
