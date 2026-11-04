"""Tests for codesearch/retrieval/rerank.py.

The cross-encoder is never downloaded here: tests either inject a fake
``scorer`` or replace the ``sentence_transformers`` module with a fake.
One slow test runs the real model (``pytest -m slow``).
"""

from __future__ import annotations

import sys
import types
from collections.abc import Sequence
from typing import Any, ClassVar

import pytest

from codesearch.config import Settings
from codesearch.factory import build_reranker
from codesearch.interfaces import Reranker
from codesearch.retrieval.rerank import (
    DEFAULT_RERANKER,
    CrossEncoderReranker,
    NoopReranker,
)
from codesearch.schemas import Chunk, ScoredChunk

QUERY = "How do I add state to a component?"


def sc(chunk_id: str, text: str | None = None, **scores: Any) -> ScoredChunk:
    chunk = Chunk(
        id=chunk_id,
        url=f"https://react.dev/{chunk_id}",
        anchor="",
        title="t",
        header_path=(),
        text=text if text is not None else f"text of {chunk_id}",
        source_path=f"{chunk_id}.md",
        start_line=0,
    )
    return ScoredChunk(chunk=chunk, **scores)


def ids(results: Sequence[ScoredChunk]) -> list[str]:
    return [r.chunk.id for r in results]


class FakeScorer:
    """Scores each (query, text) pair from a lookup on the text; records calls."""

    def __init__(self, by_text: dict[str, float]) -> None:
        self.by_text = by_text
        self.calls: list[list[tuple[str, str]]] = []

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return [self.by_text[text] for _, text in pairs]


CANDIDATES = [
    sc("a", bm25_rank=1, bm25_score=9.0, rrf_score=0.03),
    sc("b", dense_rank=1, dense_score=0.8, rrf_score=0.02),
    sc("c", bm25_rank=2, dense_rank=2, rrf_score=0.01),
]
SCORES = {"text of a": 0.1, "text of b": 2.5, "text of c": 1.0}


# =====================================================================
# NoopReranker
# =====================================================================


def test_noop_keeps_order_and_cuts_to_top_n() -> None:
    assert ids(NoopReranker().rerank(QUERY, CANDIDATES, top_n=2)) == ["a", "b"]


def test_noop_leaves_chunks_untouched() -> None:
    out = NoopReranker().rerank(QUERY, CANDIDATES, top_n=3)
    assert out == CANDIDATES
    assert all(r.rerank_score is None for r in out)


def test_noop_top_n_larger_than_input_returns_everything() -> None:
    assert ids(NoopReranker().rerank(QUERY, CANDIDATES, top_n=50)) == ["a", "b", "c"]


def test_noop_empty_input() -> None:
    assert NoopReranker().rerank(QUERY, [], top_n=5) == []


@pytest.mark.parametrize("top_n", [0, -1])
def test_noop_rejects_non_positive_top_n(top_n: int) -> None:
    with pytest.raises(ValueError):
        NoopReranker().rerank(QUERY, CANDIDATES, top_n=top_n)


def test_noop_returns_a_new_list() -> None:
    candidates = list(CANDIDATES)
    out = NoopReranker().rerank(QUERY, candidates, top_n=3)
    out.pop()
    assert len(candidates) == 3


# =====================================================================
# CrossEncoderReranker with an injected scorer
# =====================================================================


def test_reorders_by_score_descending() -> None:
    reranker = CrossEncoderReranker(scorer=FakeScorer(SCORES))
    assert ids(reranker.rerank(QUERY, CANDIDATES, top_n=3)) == ["b", "c", "a"]


def test_sets_rerank_score_as_a_plain_float() -> None:
    out = CrossEncoderReranker(scorer=FakeScorer(SCORES)).rerank(QUERY, CANDIDATES, top_n=3)
    assert [r.rerank_score for r in out] == pytest.approx([2.5, 1.0, 0.1])
    assert all(type(r.rerank_score) is float for r in out)


def test_keeps_every_earlier_score() -> None:
    """The explanation trace needs bm25/dense/rrf scores to survive reranking."""
    out = CrossEncoderReranker(scorer=FakeScorer(SCORES)).rerank(QUERY, CANDIDATES, top_n=3)
    by_id = {r.chunk.id: r for r in out}
    for original in CANDIDATES:
        new = by_id[original.chunk.id]
        assert new.chunk == original.chunk
        assert (new.bm25_rank, new.bm25_score) == (original.bm25_rank, original.bm25_score)
        assert (new.dense_rank, new.dense_score) == (original.dense_rank, original.dense_score)
        assert new.rrf_score == original.rrf_score


def test_cuts_to_top_n_after_sorting() -> None:
    out = CrossEncoderReranker(scorer=FakeScorer(SCORES)).rerank(QUERY, CANDIDATES, top_n=1)
    assert ids(out) == ["b"]


def test_top_n_larger_than_input_returns_all_sorted() -> None:
    out = CrossEncoderReranker(scorer=FakeScorer(SCORES)).rerank(QUERY, CANDIDATES, top_n=10)
    assert ids(out) == ["b", "c", "a"]


def test_scores_all_pairs_in_one_batch_in_input_order() -> None:
    scorer = FakeScorer(SCORES)
    CrossEncoderReranker(scorer=scorer).rerank(QUERY, CANDIDATES, top_n=2)
    assert scorer.calls == [[(QUERY, "text of a"), (QUERY, "text of b"), (QUERY, "text of c")]]


def test_scores_the_full_chunk_text() -> None:
    """Chunk text includes the 'Title > Section' prefix, which helps the reranker."""
    text = "useState > Usage > Adding state\n\nCall useState at the top level."
    scorer = FakeScorer({text: 1.0})
    CrossEncoderReranker(scorer=scorer).rerank(QUERY, [sc("x", text=text)], top_n=1)
    assert scorer.calls[0] == [(QUERY, text)]


def test_ties_keep_the_incoming_order() -> None:
    scorer = FakeScorer({"text of a": 1.0, "text of b": 1.0, "text of c": 1.0})
    out = CrossEncoderReranker(scorer=scorer).rerank(QUERY, CANDIDATES, top_n=3)
    assert ids(out) == ["a", "b", "c"]


def test_negative_scores_are_fine() -> None:
    """ms-marco cross-encoders return raw logits, often negative."""
    scorer = FakeScorer({"text of a": -3.0, "text of b": -0.5, "text of c": -9.0})
    out = CrossEncoderReranker(scorer=scorer).rerank(QUERY, CANDIDATES, top_n=3)
    assert ids(out) == ["b", "a", "c"]


def test_empty_input_does_not_call_the_scorer() -> None:
    scorer = FakeScorer(SCORES)
    assert CrossEncoderReranker(scorer=scorer).rerank(QUERY, [], top_n=5) == []
    assert scorer.calls == []


@pytest.mark.parametrize("top_n", [0, -2])
def test_rejects_non_positive_top_n(top_n: int) -> None:
    with pytest.raises(ValueError):
        CrossEncoderReranker(scorer=FakeScorer(SCORES)).rerank(QUERY, CANDIDATES, top_n=top_n)


def test_wrong_number_of_scores_is_an_error() -> None:
    def short_scorer(pairs: list[tuple[str, str]]) -> list[float]:
        return [1.0] * (len(pairs) - 1)

    with pytest.raises(ValueError):
        CrossEncoderReranker(scorer=short_scorer).rerank(QUERY, CANDIDATES, top_n=3)


def test_does_not_mutate_the_candidates() -> None:
    candidates = list(CANDIDATES)
    before = [c.model_copy() for c in candidates]
    CrossEncoderReranker(scorer=FakeScorer(SCORES)).rerank(QUERY, candidates, top_n=3)
    assert candidates == before
    assert all(c.rerank_score is None for c in candidates)


def test_both_rerankers_satisfy_the_protocol() -> None:
    assert isinstance(NoopReranker(), Reranker)
    assert isinstance(CrossEncoderReranker(scorer=FakeScorer({})), Reranker)


# =====================================================================
# CrossEncoderReranker loading the model (sentence_transformers faked)
# =====================================================================


class FakeCrossEncoder:
    instances: ClassVar[list[FakeCrossEncoder]] = []

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        self.predict_calls: list[list[tuple[str, str]]] = []
        FakeCrossEncoder.instances.append(self)

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.predict_calls.append(list(pairs))
        return [float(len(text)) for _, text in pairs]  # longer text = higher score


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> type[FakeCrossEncoder]:
    FakeCrossEncoder.instances = []
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return FakeCrossEncoder


def test_default_model_name() -> None:
    assert DEFAULT_RERANKER == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert CrossEncoderReranker().model_name == DEFAULT_RERANKER


def test_constructing_does_not_load_the_model(fake_st: type[FakeCrossEncoder]) -> None:
    CrossEncoderReranker("some/model")
    assert fake_st.instances == []


def test_model_is_loaded_once_and_reused(fake_st: type[FakeCrossEncoder]) -> None:
    reranker = CrossEncoderReranker("some/model")
    reranker.rerank(QUERY, CANDIDATES, top_n=2)
    reranker.rerank(QUERY, CANDIDATES, top_n=2)
    assert len(fake_st.instances) == 1
    assert fake_st.instances[0].model_name == "some/model"
    assert len(fake_st.instances[0].predict_calls) == 2


def test_uses_the_models_predict_scores(fake_st: type[FakeCrossEncoder]) -> None:
    candidates = [sc("short", text="ab"), sc("long", text="abcdef"), sc("mid", text="abcd")]
    out = CrossEncoderReranker("m").rerank(QUERY, candidates, top_n=3)
    assert ids(out) == ["long", "mid", "short"]
    assert [r.rerank_score for r in out] == pytest.approx([6.0, 4.0, 2.0])


def test_empty_input_does_not_load_the_model(fake_st: type[FakeCrossEncoder]) -> None:
    CrossEncoderReranker("m").rerank(QUERY, [], top_n=3)
    assert fake_st.instances == []


# =====================================================================
# factory.build_reranker
# =====================================================================


def test_factory_builds_noop_when_no_model_is_set() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.retrieval.reranker_model is None
    assert isinstance(build_reranker(settings), NoopReranker)


def test_factory_builds_a_cross_encoder_for_a_model_name() -> None:
    settings = Settings(_env_file=None).with_overrides(  # type: ignore[call-arg]
        retrieval={"reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    )
    reranker = build_reranker(settings)
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


# =====================================================================
# The real model (downloads ~90 MB; run with `pytest -m slow`)
# =====================================================================


@pytest.mark.slow
def test_real_cross_encoder_prefers_the_relevant_chunk() -> None:
    pytest.importorskip("sentence_transformers")
    relevant = sc(
        "rel",
        text="useState > Usage > Adding state to a component\n\n"
        "Call useState at the top level of your component to declare a state variable.",
    )
    unrelated = sc(
        "off",
        text="createPortal > Usage\n\nPortals let your components render children into "
        "a different part of the DOM.",
    )
    out = CrossEncoderReranker().rerank(QUERY, [unrelated, relevant], top_n=2)
    assert ids(out) == ["rel", "off"]
