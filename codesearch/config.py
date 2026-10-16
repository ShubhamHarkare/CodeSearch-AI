"""All configuration for CodeSearch AI, and the only module that reads env vars.

What goes here:
    - ``Settings(BaseSettings)`` (pydantic-settings) with prefix ``CODESEARCH_``,
      reading ``.env`` automatically, and nested groups:
        * ``IndexSettings``: react_dev_repo, react_dev_sha, corpus_dir (data/),
          index_dir (index/), embedding_model ("BAAI/bge-small-en-v1.5"),
          max_chars, chunk_overlap
        * ``RetrievalSettings``: mode ("dense" | "bm25" | "hybrid"), bm25_k=20,
          dense_k=20, rrf_k=60, reranker_model, rerank_top_n=5,
          abstain_threshold, history_turns=3
        * ``LLMSettings``: provider ("groq" | "ollama" | "fake"), model,
          temperature, max_tokens, timeout_s, max_retries, api_key (SecretStr)
        * ``CacheSettings``: redis_url, ttl_hours, semantic_threshold=0.95,
          semantic_max_entries
        * ``MonitoringSettings``: window_size=500, drift_threshold
        * ``AdminSettings``: webhook_secret (SecretStr), reindex_interval_hours=6
    - ``get_settings() -> Settings`` wrapped in ``functools.lru_cache``.
    - ``Settings.with_overrides(**kwargs)`` for eval configs and tests.

What it must do:
    - Validate every value at startup (fail fast with a clear message).
    - Keep secrets as ``SecretStr`` so they never appear in logs.

Must not:
    - write to ``os.environ``, or import anything else from ``codesearch``.

Tested by: tests/unit/test_config.py (defaults, env override, missing key error).
Phase: 0
"""
