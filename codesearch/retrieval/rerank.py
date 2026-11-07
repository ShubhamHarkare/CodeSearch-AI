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

from collections.abc import Callable, Sequence
from typing import Any

from codesearch.schemas import ScoredChunk

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

Scorer = Callable[[list[tuple[str, str]]], Sequence[float]]


class NoopReranker:
    """Keeps the incoming order and cuts to top_n (ablations, tests)."""

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        return list(chunks[:top_n])


class CrossEncoderReranker:
    """Re-scores (query, chunk text) pairs with a cross-encoder and keeps the best top_n."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        *,
        batch_size: int = 32,
        device: str | None = None,
        scorer: Scorer | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._scorer = scorer  # tests inject a fake; None means "use the real model"
        self._model: Any = None  # loaded on first use, then reused

    def _load(self) -> Any:
        """Load the cross-encoder once (imported here so tests never pull in PyTorch)."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """One score per pair, in the same order. predict() batches internally."""
        if self._scorer is not None:
            return [float(s) for s in self._scorer(pairs)]  # the fake gets the full list too
        model = self._load()  # loads once, cached on self
        return [float(s) for s in model.predict(pairs, batch_size=self.batch_size)]

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        if not chunks:
            return []

        pairs = [(query, c.chunk.text) for c in chunks]
        scores = self._score(pairs)

        if len(scores) != len(chunks):
            raise ValueError

        if len(scores) != len(chunks):
            raise ValueError(f"scorer returned {len(scores)} scores for {len(chunks)} chunks")

        # Sort positions by score, best first. sorted() is stable, so ties keep input order.
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        # model_copy keeps the bm25/dense/rrf scores (the explanation trace needs them).
        return [chunks[i].model_copy(update={"rerank_score": scores[i]}) for i in order[:top_n]]
