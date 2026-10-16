"""Builds concrete implementations from Settings. The only place that chooses.

What goes here:
    - ``build_embedder(settings) -> Embedder``
    - ``build_llm(settings) -> LLM`` (groq | ollama | fake)
    - ``build_reranker(settings) -> Reranker`` (cross-encoder | noop)
    - ``build_cache(settings, embedder) -> TieredCache``
    - ``load_index(settings, embedder) -> IndexHandle``
    - ``build_pipeline(settings) -> AnswerPipeline`` wiring all of the above.

What it must do:
    - Load heavy models once and share them (embedder used by retrieval,
      semantic cache and drift monitoring).

Must not:
    - do request-time work. Factory builds; pipeline runs.

Tested by: tests/integration/test_pipeline.py builds with provider="fake".
Phase: 1
"""
