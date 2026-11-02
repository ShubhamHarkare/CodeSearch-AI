"""Tests for codesearch/retrieval/hybrid.py in ``mode: dense``.

One real index is built once for the module (22 mini-corpus pages, hash
embedder, no network); every test reads it through an IndexHandle.
BM25 and hybrid modes, and rrf_fuse, get their own tests in Phase 1c.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from codesearch.config import RetrievalSettings, Settings
from codesearch.errors import IndexNotReadyError
from codesearch.index.embedder import HashEmbedder
from codesearch.index.handle import IndexHandle, IndexSnapshot
from codesearch.ingest.build import build_index
from codesearch.interfaces import Retriever
from codesearch.retrieval.hybrid import HybridRetriever
from codesearch.schemas import ScoredChunk
from tests.conftest import FakeReactDev

QUERY = "How do I update state based on the previous state?"


class SpyEmbedder(HashEmbedder):
    """HashEmbedder that records how it was called."""

    def __init__(self) -> None:
        super().__init__()
        self.query_calls: list[str] = []
        self.embed_calls = 0

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        self.embed_calls += 1
        return super().embed(texts)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        self.query_calls.append(text)
        return super().embed_query(text)


class CountingHandle(IndexHandle):
    """IndexHandle that counts snapshot() calls."""

    def __init__(self, snapshot: IndexSnapshot | None = None) -> None:
        super().__init__(snapshot)
        self.snapshot_calls = 0

    def snapshot(self) -> IndexSnapshot:
        self.snapshot_calls += 1
        return super().snapshot()


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dense")
    repo = FakeReactDev(root / "upstream")
    settings = Settings(_env_file=None).with_overrides(  # type: ignore[call-arg]
        index={
            "corpus_dir": root / "data",
            "index_dir": root / "index",
            "embedding_model": "hash",
            "react_dev_repo": repo.url,
            "react_dev_sha": repo.head,
        },
    )
    build_index(settings, HashEmbedder())
    return root / "index"


@pytest.fixture(scope="module")
def snapshot(index_dir: Path) -> IndexSnapshot:
    snap = IndexSnapshot.open(index_dir)
    assert snap is not None
    return snap


def dense(k: int = 5) -> RetrievalSettings:
    return RetrievalSettings(mode="dense", dense_k=k)


def make(snapshot: IndexSnapshot, k: int = 5) -> tuple[HybridRetriever, SpyEmbedder]:
    embedder = SpyEmbedder()
    return HybridRetriever(IndexHandle(snapshot), embedder, dense(k)), embedder


# =====================================================================
# Shape of the result
# =====================================================================


def test_returns_dense_k_scored_chunks(snapshot: IndexSnapshot) -> None:
    retriever, _ = make(snapshot, k=5)
    results = retriever.retrieve(QUERY)
    assert len(results) == 5
    assert all(isinstance(r, ScoredChunk) for r in results)


def test_ranks_start_at_one_and_are_consecutive(snapshot: IndexSnapshot) -> None:
    retriever, _ = make(snapshot, k=7)
    assert [r.dense_rank for r in retriever.retrieve(QUERY)] == list(range(1, 8))


def test_scores_are_filled_and_never_increase(snapshot: IndexSnapshot) -> None:
    retriever, _ = make(snapshot, k=10)
    scores = [r.dense_score for r in retriever.retrieve(QUERY)]
    assert all(s is not None for s in scores)
    floats = [float(s) for s in scores if s is not None]
    assert floats == sorted(floats, reverse=True)


def test_other_stage_scores_stay_empty_in_dense_mode(snapshot: IndexSnapshot) -> None:
    retriever, _ = make(snapshot)
    for r in retriever.retrieve(QUERY):
        assert r.bm25_rank is None
        assert r.bm25_score is None
        assert r.rrf_score is None
        assert r.rerank_score is None


def test_no_chunk_appears_twice(snapshot: IndexSnapshot) -> None:
    retriever, _ = make(snapshot, k=20)
    ids = [r.chunk.id for r in retriever.retrieve(QUERY)]
    assert len(ids) == len(set(ids))


# =====================================================================
# It really is dense search
# =====================================================================


def test_matches_a_direct_vector_store_query(snapshot: IndexSnapshot) -> None:
    retriever, _ = make(snapshot, k=5)
    results = retriever.retrieve(QUERY)

    direct = snapshot.vector_store.query(HashEmbedder().embed_query(QUERY), k=5)
    assert [r.chunk.id for r in results] == [chunk.id for chunk, _ in direct]
    assert [r.dense_score for r in results] == pytest.approx([score for _, score in direct])


def test_embeds_the_query_with_embed_query(snapshot: IndexSnapshot) -> None:
    """embed_query adds the bge query instruction; plain embed() would skip it."""
    retriever, embedder = make(snapshot)
    retriever.retrieve(QUERY)
    assert embedder.query_calls == [QUERY]


def test_respects_dense_k(snapshot: IndexSnapshot) -> None:
    for k in (1, 3, 12):
        retriever, _ = make(snapshot, k=k)
        assert len(retriever.retrieve(QUERY)) == k


def test_k_larger_than_the_index_returns_everything(snapshot: IndexSnapshot) -> None:
    total = snapshot.vector_store.count()
    retriever, _ = make(snapshot, k=total + 50)
    assert len(retriever.retrieve(QUERY)) == total


def test_same_query_gives_the_same_ranking(snapshot: IndexSnapshot) -> None:
    retriever, _ = make(snapshot, k=10)
    first = [r.chunk.id for r in retriever.retrieve(QUERY)]
    second = [r.chunk.id for r in retriever.retrieve(QUERY)]
    assert first == second


# =====================================================================
# Index handle
# =====================================================================


def test_reads_the_snapshot_once_per_call(snapshot: IndexSnapshot) -> None:
    handle = CountingHandle(snapshot)
    retriever = HybridRetriever(handle, SpyEmbedder(), dense())
    retriever.retrieve(QUERY)
    assert handle.snapshot_calls == 1
    retriever.retrieve(QUERY)
    assert handle.snapshot_calls == 2


def test_empty_handle_raises_index_not_ready() -> None:
    retriever = HybridRetriever(IndexHandle(), SpyEmbedder(), dense())
    with pytest.raises(IndexNotReadyError):
        retriever.retrieve(QUERY)


def test_picks_up_an_index_swapped_in_after_construction(snapshot: IndexSnapshot) -> None:
    """The retriever must ask the handle on every call, never keep its own snapshot."""
    handle = IndexHandle()
    retriever = HybridRetriever(handle, SpyEmbedder(), dense())
    with pytest.raises(IndexNotReadyError):
        retriever.retrieve(QUERY)

    handle.swap(snapshot)
    assert len(retriever.retrieve(QUERY)) == 5


def test_constructing_does_no_work(snapshot: IndexSnapshot) -> None:
    embedder = SpyEmbedder()
    handle = CountingHandle(snapshot)
    HybridRetriever(handle, embedder, dense())
    assert (handle.snapshot_calls, embedder.embed_calls, embedder.query_calls) == (0, 0, [])


# =====================================================================
# Input checks and protocol
# =====================================================================


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_is_rejected(snapshot: IndexSnapshot, query: str) -> None:
    retriever, embedder = make(snapshot)
    with pytest.raises(ValueError):
        retriever.retrieve(query)
    assert embedder.query_calls == []


def test_satisfies_the_retriever_protocol(snapshot: IndexSnapshot) -> None:
    retriever, _ = make(snapshot)
    assert isinstance(retriever, Retriever)
