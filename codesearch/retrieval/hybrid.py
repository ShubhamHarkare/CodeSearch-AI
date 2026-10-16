"""Hybrid retrieval: BM25 + dense search fused with Reciprocal Rank Fusion.

What goes here:
    - ``rrf_fuse(rankings: dict[str, list[str]], k=60) -> list[tuple[str, float]]``:
      pure function; score(d) = sum over lists of 1 / (k + rank_d). Ranks start at 1.
    - ``HybridRetriever(index_handle, embedder, settings)`` with
      ``retrieve(query) -> list[ScoredChunk]``:
        * mode "dense": dense top-k only (this + k=3 reproduces v1)
        * mode "bm25":  BM25 top-k only
        * mode "hybrid": BM25 top ``bm25_k`` + dense top ``dense_k`` -> rrf_fuse
        * fill EVERY score field on ScoredChunk (bm25_rank/score, dense_rank/
          score, rrf_score) so the explanation trace can show why each chunk ranked

What it must do:
    - Use one IndexSnapshot for the whole call (handle.snapshot()).

Tested by: tests/unit/test_rrf.py (hand-computed example, ties, item in one
list only) and tests/integration/test_pipeline.py.
Phase: 1
"""
