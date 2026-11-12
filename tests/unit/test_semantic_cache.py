"""Tests for codesearch/cache/semantic.py: reusing an answer for a paraphrased question.

The exact tier only matches after normalising case and punctuation. This tier embeds
the question and reuses a stored answer when a stored question is very similar
(cosine >= threshold). Because "How do I use useState?" and "How do I use
useReducer?" can look alike to an embedding model, a hit also needs the same top-1
retrieved chunk ("top1"), so the cache sits after retrieval and saves the expensive
LLM call.

A TableEmbedder maps each test question to a hand-picked vector, so similarities are
exact and known.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from codesearch.cache.semantic import SemanticCache
from codesearch.schemas import Answer

VERSIONS = {"index_version": "v1", "prompt_version": "1", "model": "gemma3:4b"}


def unit(*values: float) -> NDArray[np.float32]:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def at_cosine(c: float) -> NDArray[np.float32]:
    """A unit vector whose cosine with [1, 0, 0] is exactly c."""
    return np.array([c, math.sqrt(1 - c * c), 0.0], dtype=np.float32)


class TableEmbedder:
    """Embedder with fixed vectors per text; counts calls."""

    model_name = "table"

    def __init__(self, table: dict[str, NDArray[np.float32]]) -> None:
        self.table = table
        self.calls: list[str] = []

    @property
    def dim(self) -> int:
        return 3

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        return np.stack([self.embed_query(t) for t in texts])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        self.calls.append(text)
        return self.table[text]


Q = "How do I use useState?"
TABLE = {
    Q: unit(1, 0, 0),
    "How can I add state to a function component?": at_cosine(0.97),  # paraphrase
    "How do I use useReducer?": at_cosine(0.96),  # similar wording, different topic
    "What is JSX?": at_cosine(0.50),  # unrelated
    "zero": np.zeros(3, dtype=np.float32),
}
PARAPHRASE = "How can I add state to a function component?"


def answer(text: str = "Use useState [1].") -> Answer:
    return Answer(
        text=text,
        citations=[],
        abstained=False,
        trace_id="t",
        index_version="v1",
        prompt_version="1",
    )


@pytest.fixture
def embedder() -> TableEmbedder:
    return TableEmbedder(TABLE)


@pytest.fixture
def cache(embedder: TableEmbedder) -> SemanticCache:
    c = SemanticCache(embedder, threshold=0.95)
    c.add(Q, answer(), top1="useState#usage", **VERSIONS)
    return c


# =====================================================================
# Hits and misses
# =====================================================================


def test_empty_cache_misses(embedder: TableEmbedder) -> None:
    assert SemanticCache(embedder).lookup(Q, top1="useState#usage", **VERSIONS) is None


def test_same_question_hits(cache: SemanticCache) -> None:
    assert cache.lookup(Q, top1="useState#usage", **VERSIONS) == answer()


def test_paraphrase_above_threshold_hits(cache: SemanticCache) -> None:
    assert cache.lookup(PARAPHRASE, top1="useState#usage", **VERSIONS) == answer()


def test_unrelated_question_misses(cache: SemanticCache) -> None:
    assert cache.lookup("What is JSX?", top1="useState#usage", **VERSIONS) is None


def test_similar_wording_with_a_different_top_source_misses(cache: SemanticCache) -> None:
    """useReducer looks like useState to the embedder (0.96 > 0.95), but retrieval
    found a different best chunk, so the stored answer is not reused."""
    assert cache.lookup("How do I use useReducer?", top1="useReducer#usage", **VERSIONS) is None


def test_same_vector_but_different_top_source_misses(cache: SemanticCache) -> None:
    assert cache.lookup(Q, top1="something-else", **VERSIONS) is None


def test_similarity_exactly_at_the_threshold_hits(embedder: TableEmbedder) -> None:
    threshold = float(np.dot(TABLE[Q], TABLE[PARAPHRASE]))
    cache = SemanticCache(embedder, threshold=threshold)
    cache.add(Q, answer(), top1="s", **VERSIONS)
    assert cache.lookup(PARAPHRASE, top1="s", **VERSIONS) is not None


def test_just_below_the_threshold_misses(embedder: TableEmbedder) -> None:
    cache = SemanticCache(embedder, threshold=0.98)  # the paraphrase is at 0.97
    cache.add(Q, answer(), top1="s", **VERSIONS)
    assert cache.lookup(PARAPHRASE, top1="s", **VERSIONS) is None


def test_the_most_similar_stored_question_wins(embedder: TableEmbedder) -> None:
    cache = SemanticCache(embedder, threshold=0.9)
    cache.add("How do I use useReducer?", answer("reducer"), top1="s", **VERSIONS)  # 0.96 to Q
    cache.add(PARAPHRASE, answer("paraphrase"), top1="s", **VERSIONS)  # 0.97 to Q
    assert cache.lookup(Q, top1="s", **VERSIONS) == answer("paraphrase")


@pytest.mark.parametrize("field", ["index_version", "prompt_version", "model"])
def test_a_different_version_never_hits(cache: SemanticCache, field: str) -> None:
    """After a re-index, prompt change or model switch, old answers are stale."""
    changed = {**VERSIONS, field: "other"}
    assert cache.lookup(Q, top1="useState#usage", **changed) is None


def test_a_zero_vector_is_a_miss_not_a_crash(cache: SemanticCache) -> None:
    assert cache.lookup("zero", top1="useState#usage", **VERSIONS) is None


def test_adding_a_zero_vector_is_ignored(embedder: TableEmbedder) -> None:
    cache = SemanticCache(embedder)
    cache.add("zero", answer(), top1="s", **VERSIONS)
    assert len(cache) == 0


# =====================================================================
# Size, replacement, embedding calls
# =====================================================================


def test_oldest_entry_is_evicted_beyond_max_entries(embedder: TableEmbedder) -> None:
    cache = SemanticCache(embedder, threshold=0.99, max_entries=2)
    cache.add(Q, answer("q"), top1="s", **VERSIONS)
    cache.add("What is JSX?", answer("jsx"), top1="s", **VERSIONS)
    cache.add(PARAPHRASE, answer("p"), top1="s", **VERSIONS)
    assert len(cache) == 2
    assert cache.lookup(Q, top1="s", **VERSIONS) is None  # evicted (and 0.97 < 0.99)
    assert cache.lookup("What is JSX?", top1="s", **VERSIONS) == answer("jsx")


def test_re_adding_the_same_question_replaces_it(embedder: TableEmbedder) -> None:
    cache = SemanticCache(embedder)
    cache.add(Q, answer("old"), top1="s", **VERSIONS)
    # same question after normalize(); its vector is the same as Q's
    cache.add("  how do i use usestate  ", answer("new"), top1="s", vector=TABLE[Q], **VERSIONS)
    assert len(cache) == 1
    assert cache.lookup(Q, top1="s", **VERSIONS) == answer("new")


def test_the_same_question_under_new_versions_is_a_separate_entry(
    embedder: TableEmbedder,
) -> None:
    cache = SemanticCache(embedder)
    cache.add(Q, answer("v1"), top1="s", **VERSIONS)
    cache.add(Q, answer("v2"), top1="s", **{**VERSIONS, "index_version": "v2"})
    assert len(cache) == 2


def test_lookup_embeds_the_query_once(cache: SemanticCache, embedder: TableEmbedder) -> None:
    embedder.calls.clear()
    cache.lookup(PARAPHRASE, top1="useState#usage", **VERSIONS)
    assert embedder.calls == [PARAPHRASE]


def test_add_can_reuse_a_vector_it_was_given(embedder: TableEmbedder) -> None:
    """The pipeline already embedded the question for retrieval; don't do it twice."""
    cache = SemanticCache(embedder)
    cache.add(Q, answer(), top1="s", vector=TABLE[Q], **VERSIONS)
    assert embedder.calls == []
    assert cache.lookup(Q, top1="s", **VERSIONS) == answer()


def test_clear(cache: SemanticCache) -> None:
    cache.clear()
    assert len(cache) == 0
    assert cache.lookup(Q, top1="useState#usage", **VERSIONS) is None


# =====================================================================
# Settings
# =====================================================================


def test_defaults(embedder: TableEmbedder) -> None:
    cache = SemanticCache(embedder)
    assert (cache.threshold, cache.max_entries) == (0.95, 1000)


@pytest.mark.parametrize("threshold", [0.0, -0.5, 1.01])
def test_threshold_must_be_in_zero_one(embedder: TableEmbedder, threshold: float) -> None:
    with pytest.raises(ValueError):
        SemanticCache(embedder, threshold=threshold)


def test_max_entries_must_be_positive(embedder: TableEmbedder) -> None:
    with pytest.raises(ValueError):
        SemanticCache(embedder, max_entries=0)
