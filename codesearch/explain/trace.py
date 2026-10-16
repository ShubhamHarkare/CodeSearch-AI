"""Records how each answer was produced and stores it by trace_id.

What goes here:
    - ``TraceBuilder``: the pipeline calls ``add_candidates()``,
      ``set_used()``, ``set_sentences()``, ``set_abstain()``, ``time(step)``
      (context manager) and finally ``build() -> ExplanationTrace``.
    - ``TraceStore(db_path)``: SQLite; ``save(trace)``, ``get(trace_id)``
      (raises TraceNotFoundError), ``recent(n)``; stores the trace as JSON.
    - per-source "why" data: BM25 term contributions (index/bm25.py) attached
      to each used chunk.

Served by: GET /v1/explain/{trace_id}; rendered in the UI's
"Why this answer?" panel.

Tested by: tests/unit/test_trace.py (round-trip; timings recorded).
Phase: 2
"""
