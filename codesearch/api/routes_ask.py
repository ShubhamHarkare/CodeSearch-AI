"""User-facing endpoints. Each handler: validate, call the pipeline, shape the response.

Endpoints:
    - ``POST /v1/ask`` -> AskResponse (JSON).
    - ``POST /v1/ask/stream`` -> text/event-stream; events in this order:
      retrieval (sources + scores), token (repeated), citations, groundedness,
      done (timings, trace_id); error if something fails mid-stream.
    - ``GET /v1/explain/{trace_id}`` -> the stored ExplanationTrace.
    - ``POST /v1/explain/{trace_id}/deep`` -> leave-one-out attribution
      (rate-limit it: it costs k LLM calls).
    - ``POST /v1/feedback`` -> 204.

Must not:
    - contain retrieval or LLM logic.

Tested by: tests/integration/test_api.py (JSON answer, SSE event order,
404 on unknown trace, feedback stored).
Phase: 1 (ask), 2 (explain), 3 (stream, feedback)
"""
