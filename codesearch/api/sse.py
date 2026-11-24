"""Server-sent events helpers.

What goes here:
    - ``format_event(event: StreamEvent) -> str``: ``event: <type>\ndata: <json>\n\n``.
    - ``async stream_response(events, request) -> AsyncIterator[str]``:
        * forwards pipeline events as SSE frames
        * checks ``await request.is_disconnected()`` and stops early, which
          closes the LLM stream so you stop paying for unread tokens
        * sends a heartbeat comment every ~15 s on long gaps
        * turns exceptions into a final ``error`` event with the trace_id

Tested by: tests/unit/test_sse.py (frame format) and the API stream test.
Phase: 3
"""

import json
from collections.abc import AsyncGenerator
from contextlib import aclosing

from fastapi.requests import Request

from codesearch.errors import CodeSearchError
from codesearch.log import get_logger
from codesearch.schemas import StreamEvent

UNEXPECTED_ERROR = "An unexpected error has occured"

log = get_logger(__name__)


def format_event(event: StreamEvent) -> str:
    return f"event: {event.type}\ndata: {json.dumps(event.data)}\n\n"


async def stream_response(
    events: AsyncGenerator[StreamEvent], request: Request, trace_id: str
) -> AsyncGenerator[str]:
    try:
        async with aclosing(events) as source:
            async for event in source:
                if await request.is_disconnected():
                    return  # User has closed the tab

                yield format_event(event)

    except CodeSearchError as e:
        # The user only sees the public message; the real reason goes to the log.
        log.warning("stream.error", trace_id=trace_id, error=type(e).__name__, detail=str(e))
        yield format_event(
            StreamEvent(type="error", data={"error": e.public_message, "trace_id": trace_id})
        )

    except Exception as e:
        log.exception("stream.failed", trace_id=trace_id, error=type(e).__name__)
        yield format_event(
            StreamEvent(type="error", data={"error": UNEXPECTED_ERROR, "trace_id": trace_id})
        )
