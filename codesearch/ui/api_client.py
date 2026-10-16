"""HTTP client the Streamlit UI uses to talk to the API.

What goes here:
    - ``ApiClient(base_url, timeout_s)`` (httpx):
        * ``ask(question, history) -> dict``
        * ``ask_stream(question, history) -> Iterator[StreamEvent]``: parse SSE
          frames into typed events
        * ``explain(trace_id)``, ``deep_explain(trace_id)``
        * ``send_feedback(trace_id, vote, comment)``
        * ``ready() -> dict`` and ``metrics_summary() -> dict`` for the sidebar
    - Clear error messages when the API is down (show them in the UI, don't crash).

Must not:
    - import anything from codesearch except schemas.py.

Tested by: tests/unit/test_api_client.py with httpx.MockTransport.
Phase: 1, 3 (stream)
"""
