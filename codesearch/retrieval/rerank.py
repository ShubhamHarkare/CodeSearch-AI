"""Reorders retrieval candidates with a cross-encoder.

What goes here:
    - ``CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")``
      with ``rerank(query, chunks, top_n) -> list[ScoredChunk]``: score each
      (query, chunk.text) pair in one batch, set ``rerank_score``, sort, cut to top_n.
    - ``NoopReranker``: keeps the fused order and cuts to top_n (ablations, tests).

What it must do:
    - Load the model once; keep latency in the trace's timings.

Tested by: tests/unit/test_rerank.py (Noop keeps order; a fake scorer reorders).
Phase: 1
"""
