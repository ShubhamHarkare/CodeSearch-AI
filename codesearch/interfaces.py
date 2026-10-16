"""Contracts (typing.Protocol) that swappable parts must satisfy.

What goes here:
    - ``Embedder``: ``model_name: str``, ``dim: int``,
      ``embed(texts: list[str]) -> np.ndarray`` (L2-normalised, shape (n, dim)).
    - ``LLM``: ``generate(prompt: str) -> str`` and
      ``stream(prompt: str) -> AsyncIterator[str]``.
    - ``Reranker``: ``rerank(query, chunks: list[ScoredChunk], top_n) -> list[ScoredChunk]``.
    - ``Cache``: ``get(key) -> Answer | None``, ``set(key, answer) -> None``.
    - ``Retriever``: ``retrieve(query) -> list[ScoredChunk]``.

Why: pipeline.py depends on these protocols, not on Groq, Chroma or Redis, so
tests can pass in FakeLLM / HashEmbedder and the eval harness can swap parts.

Must not:
    - contain implementations.

Phase: 0
"""
