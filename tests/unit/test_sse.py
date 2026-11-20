"""Tests for codesearch/api/sse.py: the SSE frame format and the stream loop.

stream_response is tested with a fake request (is_disconnected) and fake event
generators, so no server runs. The HTTP route is tested in tests/integration/test_api.py.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from codesearch.api.sse import format_event, stream_response
from codesearch.errors import IndexNotReadyError, LLMUnavailableError
from codesearch.schemas import StreamEvent


def parse(frame: str) -> tuple[str, Any]:
    """'event: X\\ndata: Y\\n\\n' -> ("X", json.loads(Y)); fails on anything else."""
    assert frame.endswith("\n\n"), "a frame ends with a blank line"
    lines = frame[:-2].split("\n")
    assert len(lines) == 2, f"expected exactly an event line and a data line: {lines}"
    assert lines[0].startswith("event: ") and lines[1].startswith("data: ")
    return lines[0][len("event: ") :], json.loads(lines[1][len("data: ") :])


class FakeRequest:
    """is_disconnected() turns True after `connected_for` checks."""

    def __init__(self, connected_for: int = 10_000) -> None:
        self.checks = 0
        self.connected_for = connected_for

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self.checks > self.connected_for


class Events:
    """A fake pipeline stream: yields the given events, then optionally raises."""

    def __init__(self, events: list[StreamEvent], error: Exception | None = None) -> None:
        self.events = events
        self.error = error
        self.closed = False

    async def __call__(self) -> AsyncGenerator[StreamEvent]:
        try:
            for event in self.events:
                yield event
            if self.error is not None:
                raise self.error
        finally:
            self.closed = True


async def frames(events: Events, request: FakeRequest | None = None) -> list[tuple[str, Any]]:
    out = [f async for f in stream_response(events(), request or FakeRequest(), "trace-1")]
    return [parse(f) for f in out]


TOKENS = [
    StreamEvent(type="retrieval", data=[{"number": 1, "url": "https://react.dev/a"}]),
    StreamEvent(type="token", data="Use"),
    StreamEvent(type="token", data=" useState"),
    StreamEvent(type="done", data={"text": "Use useState", "trace_id": "trace-1"}),
]


# =====================================================================
# format_event
# =====================================================================


def test_format_event_writes_an_event_line_and_a_json_data_line() -> None:
    assert format_event(StreamEvent(type="token", data="Use")) == 'event: token\ndata: "Use"\n\n'


def test_format_event_round_trips_a_dict() -> None:
    data = {"text": "Use useState [1].", "citations": [{"number": 1}], "abstained": False}
    assert parse(format_event(StreamEvent(type="done", data=data))) == ("done", data)


def test_a_newline_inside_a_token_cannot_break_the_frame() -> None:
    """Code blocks contain newlines; a raw one would end the data line early."""
    frame = format_event(StreamEvent(type="token", data="```js\nconst x = 1;\n```"))
    assert parse(frame) == ("token", "```js\nconst x = 1;\n```")


# =====================================================================
# stream_response
# =====================================================================


async def test_every_event_becomes_a_frame_in_order() -> None:
    got = await frames(Events(TOKENS))
    assert [kind for kind, _ in got] == ["retrieval", "token", "token", "done"]
    assert got[1][1] == "Use"


async def test_it_stops_and_closes_the_pipeline_when_the_client_leaves() -> None:
    events = Events(TOKENS)
    got = await frames(events, FakeRequest(connected_for=2))
    assert [kind for kind, _ in got] == ["retrieval", "token"]
    assert events.closed  # closing it is what stops the LLM upstream


async def test_the_pipeline_is_closed_after_a_normal_finish_too() -> None:
    events = Events(TOKENS)
    await frames(events)
    assert events.closed


async def test_a_codesearch_error_mid_stream_ends_with_an_error_event() -> None:
    events = Events(TOKENS[:2], error=LLMUnavailableError("ollama crashed at 10.0.0.5"))
    got = await frames(events)
    assert [kind for kind, _ in got] == ["retrieval", "token", "error"]
    assert got[-1][1] == {"error": LLMUnavailableError.public_message, "trace_id": "trace-1"}
    assert events.closed


async def test_an_error_before_any_event_is_the_only_frame() -> None:
    got = await frames(Events([], error=IndexNotReadyError()))
    assert got == [("error", {"error": IndexNotReadyError.public_message, "trace_id": "trace-1"})]


async def test_an_unexpected_error_never_leaks_its_text() -> None:
    got = await frames(Events(TOKENS[:1], error=RuntimeError("password=hunter2")))
    kind, data = got[-1]
    assert kind == "error"
    assert data["trace_id"] == "trace-1"
    assert "hunter2" not in data["error"]
