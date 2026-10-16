"""Near-duplicate question tier: reuse an answer for a paraphrased question.

What goes here:
    - ``SemanticCache(embedder, threshold=0.95, max_entries=2000)``:
        * ``lookup(query, index_version, prompt_version) -> Answer | None``:
          embed the query; brute-force cosine against stored query vectors
          (numpy matrix, fine at this size) filtered to the same versions
        * a hit needs cosine >= threshold AND the same top-1 source as a fresh
          retrieval would give (pass that in), to avoid "useState" vs
          "useReducer" false hits
        * ``add(query, vector, answer)``; evict oldest beyond max_entries
    - Optionally persist vectors in Redis; in-process is fine to start.

Tested by: tests/unit/test_semantic_cache.py with HashEmbedder (exact
paraphrase hits; different topic misses; version mismatch misses).
Phase: 1
"""
