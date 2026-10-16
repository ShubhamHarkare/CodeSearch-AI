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
