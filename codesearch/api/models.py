"""Request and response bodies for the HTTP API.

Kept separate from schemas.py so the public API contract can change without
touching internals (and vice versa).

What goes here:
    - ``AskRequest``: question (1..1000 chars), history (list of {role, content},
      max 10), bypass_cache.
    - ``AskResponse``: answer, citations, abstained, confidence, groundedness,
      cache_tier, timings, trace_id.
    - ``FeedbackRequest``: trace_id, vote (+1 / -1), comment (optional, max 500).
    - ``IndexInfo``, ``HealthStatus`` (status, components, versions, image_sha).

Phase: 1
"""
