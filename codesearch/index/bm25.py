"""Keyword search with per-term explanations.

What goes here:
    - ``tokenize(text) -> list[str]``: lowercase words, AND keep API names whole
      plus their camelCase parts (``useLayoutEffect`` -> ``uselayouteffect``,
      ``use``, ``layout``, ``effect``); drop markdown punctuation.
    - ``BM25Index.build(chunks) -> BM25Index`` (rank_bm25.BM25Okapi).
    - ``search(query, k) -> list[tuple[str, float]]`` (chunk_id, score).
    - ``term_contributions(query, chunk_id) -> dict[str, float]``: how much each
      query term added to that chunk's score (for the "why this source" panel).
    - ``save(path)`` / ``load(path)`` so the API starts without rebuilding.

Why: exact API names are where dense embeddings are weakest; this is half of
hybrid retrieval and one row of the ablation table.

Tested by: tests/unit/test_bm25.py (exact API name ranks first, camelCase
split, contributions sum to the score).
Phase: 1 (search), 2 (term_contributions)
"""
