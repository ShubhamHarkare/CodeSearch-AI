"""Tests for codesearch/retrieval/abstain.py.

The abstain gate looks at the best retrieval score and refuses to answer when
it is too low. Which score: the cross-encoder's rerank_score when the chunks
were reranked, otherwise the dense cosine similarity. BM25 and RRF scores are
not on a meaningful absolute scale, so they are never used.
"""

from __future__ import annotations

from typing import Any

import pytest

from codesearch.retrieval.abstain import ABSTAIN_MESSAGE, AbstainDecision, decide, top_signal
from codesearch.schemas import Chunk, ScoredChunk


def sc(chunk_id: str = "c", **scores: Any) -> ScoredChunk:
    chunk = Chunk(
        id=chunk_id,
        url="https://react.dev/learn",
        anchor="",
        title="t",
        header_path=(),
        text="x",
        source_path="x.md",
        start_line=0,
    )
    return ScoredChunk(chunk=chunk, **scores)


# =====================================================================
# top_signal: which score the gate looks at
# =====================================================================


def test_signal_is_the_best_dense_score_without_reranking() -> None:
    chunks = [sc("a", dense_score=0.62), sc("b", dense_score=0.81), sc("c", dense_score=0.40)]
    assert top_signal(chunks) == ("dense", pytest.approx(0.81))


def test_rerank_score_wins_over_dense_when_present() -> None:
    chunks = [
        sc("a", dense_score=0.9, rerank_score=-2.0),
        sc("b", dense_score=0.5, rerank_score=3.5),
    ]
    assert top_signal(chunks) == ("rerank", pytest.approx(3.5))


def test_signal_ignores_chunks_missing_that_score() -> None:
    """In hybrid mode, BM25-only chunks have no dense_score."""
    chunks = [sc("bm25only", bm25_score=12.0, rrf_score=0.03), sc("d", dense_score=0.55)]
    assert top_signal(chunks) == ("dense", pytest.approx(0.55))


def test_negative_rerank_scores_are_compared_correctly() -> None:
    chunks = [sc("a", rerank_score=-7.0), sc("b", rerank_score=-1.5)]
    assert top_signal(chunks) == ("rerank", pytest.approx(-1.5))


def test_bm25_and_rrf_scores_are_never_used() -> None:
    assert top_signal([sc(bm25_score=30.0, rrf_score=0.05)]) is None


def test_no_chunks_means_no_signal() -> None:
    assert top_signal([]) is None


def test_signal_value_is_a_plain_float() -> None:
    signal = top_signal([sc(dense_score=0.7)])
    assert signal is not None
    assert type(signal[1]) is float


# =====================================================================
# decide
# =====================================================================


def test_answers_above_the_threshold() -> None:
    d = decide([sc(dense_score=0.8)], threshold=0.6)
    assert d == AbstainDecision(
        abstain=False, reason=None, top_score=pytest.approx(0.8), score_kind="dense"
    )


def test_answers_exactly_at_the_threshold() -> None:
    assert decide([sc(dense_score=0.6)], threshold=0.6).abstain is False


def test_abstains_below_the_threshold() -> None:
    d = decide([sc(dense_score=0.59)], threshold=0.6)
    assert d.abstain is True
    assert d.reason == "below_threshold"
    assert d.top_score == pytest.approx(0.59)
    assert d.score_kind == "dense"


def test_abstains_when_nothing_was_retrieved() -> None:
    d = decide([], threshold=0.6)
    assert d == AbstainDecision(
        abstain=True, reason="no_candidates", top_score=None, score_kind=None
    )


def test_uses_the_best_chunk_not_the_first() -> None:
    chunks = [sc("a", dense_score=0.3), sc("b", dense_score=0.9)]
    assert decide(chunks, threshold=0.6).abstain is False


def test_rerank_threshold_works_on_logits() -> None:
    chunks = [sc(dense_score=0.95, rerank_score=-4.0)]
    d = decide(chunks, threshold=0.0)
    assert (d.abstain, d.score_kind, d.top_score) == (True, "rerank", pytest.approx(-4.0))


def test_chunks_without_a_usable_score_are_an_error() -> None:
    """BM25-only results can't be judged; that's a configuration mistake, not a refusal."""
    with pytest.raises(ValueError):
        decide([sc(bm25_score=10.0)], threshold=0.5)


def test_decide_does_not_change_the_chunks() -> None:
    chunks = [sc("a", dense_score=0.4), sc("b", dense_score=0.9)]
    before = list(chunks)
    decide(chunks, threshold=0.5)
    assert chunks == before


def test_decision_is_frozen() -> None:
    d = decide([sc(dense_score=0.8)], threshold=0.5)
    with pytest.raises((AttributeError, TypeError)):
        d.abstain = True  # type: ignore[misc]


# =====================================================================
# The canned reply
# =====================================================================


def test_abstain_message_points_to_the_react_docs() -> None:
    assert "react.dev" in ABSTAIN_MESSAGE
    assert len(ABSTAIN_MESSAGE) < 300
