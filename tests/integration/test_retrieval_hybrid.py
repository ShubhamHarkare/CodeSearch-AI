"""Tests for HybridRetriever in ``mode: bm25`` and ``mode: hybrid``.

Same setup as test_retrieval_dense.py: one real index built once from the
22 mini-corpus pages with the hash embedder. Expected results are computed
from the index's own BM25 / vector store / rrf_fuse, so the tests check the
wiring, not the quality of the hash embedder.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from codesearch.config import RetrievalSettings, Settings
from codesearch.errors import IndexNotReadyError
from codesearch.index.embedder import HashEmbedder
from codesearch.index.handle import IndexHandle, IndexSnapshot
from codesearch.ingest.build import build_index
from codesearch.retrieval.hybrid import HybridRetriever, rrf_fuse
from codesearch.schemas import ScoredChunk
from tests.conftest import FakeReactDev

QUERY = "How do I update state based on the previous state?"
KEYWORD_QUERY = "createPortal"
NO_MATCH_QUERY = "zzqx wvvk"  # tokens that appear nowhere in the corpus


class SpyEmbedder(HashEmbedder):
    """HashEmbedder that counts query embeddings and raw embed() calls separately."""

    def __init__(self) -> None:
        super().__init__()
        self.query_calls = 0
        self.embed_calls = 0

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        self.embed_calls += 1
        return super().embed(texts)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        self.query_calls += 1
        return super().embed_query(text)

    @property
    def calls(self) -> int:
        """Query embeddings asked for (embed_query uses embed() inside, so count those)."""
        return self.query_calls


class CountingHandle(IndexHandle):
    def __init__(self, snapshot: IndexSnapshot | None = None) -> None:
        super().__init__(snapshot)
        self.snapshot_calls = 0

    def snapshot(self) -> IndexSnapshot:
        self.snapshot_calls += 1
        return super().snapshot()


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> IndexSnapshot:
    root = tmp_path_factory.mktemp("hybrid")
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
    snap = IndexSnapshot.open(root / "index")
    assert snap is not None
    return snap


def make(
    snapshot: IndexSnapshot, mode: str, **kw: int
) -> tuple[HybridRetriever, SpyEmbedder, RetrievalSettings]:
    settings = RetrievalSettings(mode=mode, **kw)  # type: ignore[arg-type]
    embedder = SpyEmbedder()
    return HybridRetriever(IndexHandle(snapshot), embedder, settings), embedder, settings


def chunk_ids(results: list[ScoredChunk]) -> list[str]:
    return [r.chunk.id for r in results]


# =====================================================================
# mode: bm25
# =====================================================================


def test_bm25_matches_a_direct_bm25_search(snapshot: IndexSnapshot) -> None:
    retriever, _, _ = make(snapshot, "bm25", bm25_k=8)
    results = retriever.retrieve(QUERY)
    direct = snapshot.bm25.search(QUERY, k=8)
    assert chunk_ids(results) == [doc_id for doc_id, _ in direct]
    assert [r.bm25_score for r in results] == pytest.approx([score for _, score in direct])


def test_bm25_ranks_start_at_one_and_scores_never_increase(snapshot: IndexSnapshot) -> None:
    retriever, _, _ = make(snapshot, "bm25", bm25_k=10)
    results = retriever.retrieve(QUERY)
    assert [r.bm25_rank for r in results] == list(range(1, len(results) + 1))
    values = [float(r.bm25_score) for r in results if r.bm25_score is not None]
    assert len(values) == len(results)
    assert values == sorted(values, reverse=True)


def test_bm25_respects_bm25_k(snapshot: IndexSnapshot) -> None:
    for k in (1, 4, 15):
        retriever, _, _ = make(snapshot, "bm25", bm25_k=k)
        assert len(retriever.retrieve(QUERY)) == k


def test_bm25_leaves_dense_and_rrf_fields_empty(snapshot: IndexSnapshot) -> None:
    retriever, _, _ = make(snapshot, "bm25")
    for r in retriever.retrieve(QUERY):
        assert (r.dense_rank, r.dense_score, r.rrf_score, r.rerank_score) == (None,) * 4


def test_bm25_never_calls_the_embedder(snapshot: IndexSnapshot) -> None:
    retriever, embedder, _ = make(snapshot, "bm25")
    retriever.retrieve(QUERY)
    assert (embedder.query_calls, embedder.embed_calls) == (0, 0)


def test_bm25_returns_full_chunks(snapshot: IndexSnapshot) -> None:
    retriever, _, _ = make(snapshot, "bm25", bm25_k=3)
    results = retriever.retrieve(QUERY)
    stored = snapshot.vector_store.get(chunk_ids(results))
    assert [r.chunk for r in results] == stored


def test_bm25_finds_an_exact_api_name(snapshot: IndexSnapshot) -> None:
    retriever, _, _ = make(snapshot, "bm25", bm25_k=1)
    (top,) = retriever.retrieve(KEYWORD_QUERY)
    assert top.chunk.source_path == "reference/react-dom/createPortal.md"


def test_bm25_with_no_matching_terms_returns_nothing(snapshot: IndexSnapshot) -> None:
    retriever, _, _ = make(snapshot, "bm25")
    assert retriever.retrieve(NO_MATCH_QUERY) == []


# =====================================================================
# mode: hybrid
# =====================================================================


def expected_fusion(
    snapshot: IndexSnapshot, query: str, s: RetrievalSettings
) -> list[tuple[str, float]]:
    bm25_ids = [doc_id for doc_id, _ in snapshot.bm25.search(query, k=s.bm25_k)]
    dense = snapshot.vector_store.query(HashEmbedder().embed_query(query), k=s.dense_k)
    return rrf_fuse({"bm25": bm25_ids, "dense": [c.id for c, _ in dense]}, k=s.rrf_k)


def test_hybrid_order_is_the_rrf_fusion_of_both_lists(snapshot: IndexSnapshot) -> None:
    retriever, _, s = make(snapshot, "hybrid", bm25_k=10, dense_k=10)
    results = retriever.retrieve(QUERY)
    fused = expected_fusion(snapshot, QUERY, s)
    assert chunk_ids(results) == [doc_id for doc_id, _ in fused]
    assert [r.rrf_score for r in results] == pytest.approx([score for _, score in fused])


def test_hybrid_returns_the_union_of_both_lists(snapshot: IndexSnapshot) -> None:
    retriever, _, _ = make(snapshot, "hybrid", bm25_k=10, dense_k=10)
    results = retriever.retrieve(QUERY)
    bm25_ids = {d for d, _ in snapshot.bm25.search(QUERY, k=10)}
    dense_ids = {
        c.id for c, _ in snapshot.vector_store.query(HashEmbedder().embed_query(QUERY), k=10)
    }
    assert set(chunk_ids(results)) == bm25_ids | dense_ids
    assert len(results) == len(bm25_ids | dense_ids)  # no duplicates


def test_hybrid_fills_every_score_from_the_list_a_chunk_came_from(snapshot: IndexSnapshot) -> None:
    retriever, _, _ = make(snapshot, "hybrid", bm25_k=10, dense_k=10)
    results = retriever.retrieve(QUERY)

    bm25 = {
        d: (rank, score) for rank, (d, score) in enumerate(snapshot.bm25.search(QUERY, k=10), 1)
    }
    dense_hits = snapshot.vector_store.query(HashEmbedder().embed_query(QUERY), k=10)
    dense = {c.id: (rank, score) for rank, (c, score) in enumerate(dense_hits, 1)}

    for r in results:
        assert r.rrf_score is not None
        assert r.rerank_score is None
        if r.chunk.id in bm25:
            assert (r.bm25_rank, r.bm25_score) == (
                bm25[r.chunk.id][0],
                pytest.approx(bm25[r.chunk.id][1]),
            )
        else:
            assert (r.bm25_rank, r.bm25_score) == (None, None)
        if r.chunk.id in dense:
            assert (r.dense_rank, r.dense_score) == (
                dense[r.chunk.id][0],
                pytest.approx(dense[r.chunk.id][1]),
            )
        else:
            assert (r.dense_rank, r.dense_score) == (None, None)


def test_hybrid_uses_rrf_k_from_settings(snapshot: IndexSnapshot) -> None:
    low, _, _ = make(snapshot, "hybrid", rrf_k=0)
    high, _, _ = make(snapshot, "hybrid", rrf_k=60)
    top_low = low.retrieve(QUERY)[0].rrf_score
    top_high = high.retrieve(QUERY)[0].rrf_score
    assert top_low is not None and top_high is not None
    assert top_low > top_high


def test_hybrid_respects_both_k_settings(snapshot: IndexSnapshot) -> None:
    retriever, _, _ = make(snapshot, "hybrid", bm25_k=2, dense_k=3)
    results = retriever.retrieve(QUERY)
    assert 3 <= len(results) <= 5
    assert sum(r.bm25_rank is not None for r in results) == 2
    assert sum(r.dense_rank is not None for r in results) == 3


def test_hybrid_embeds_the_query_once_and_reads_one_snapshot(snapshot: IndexSnapshot) -> None:
    handle = CountingHandle(snapshot)
    embedder = SpyEmbedder()
    HybridRetriever(handle, embedder, RetrievalSettings(mode="hybrid")).retrieve(QUERY)
    assert embedder.calls == 1
    assert handle.snapshot_calls == 1


def test_hybrid_keyword_query_puts_the_api_page_near_the_top(snapshot: IndexSnapshot) -> None:
    retriever, _, _ = make(snapshot, "hybrid")
    top3 = {r.chunk.source_path for r in retriever.retrieve(KEYWORD_QUERY)[:3]}
    assert "reference/react-dom/createPortal.md" in top3


def test_hybrid_with_no_bm25_matches_falls_back_to_dense_order(snapshot: IndexSnapshot) -> None:
    hybrid, _, _ = make(snapshot, "hybrid", dense_k=5)
    dense, _, _ = make(snapshot, "dense", dense_k=5)
    h = hybrid.retrieve(NO_MATCH_QUERY)
    assert chunk_ids(h) == chunk_ids(dense.retrieve(NO_MATCH_QUERY))
    assert all(r.bm25_rank is None and r.rrf_score is not None for r in h)


# =====================================================================
# Shared rules across modes
# =====================================================================


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_is_rejected_in_every_mode(
    snapshot: IndexSnapshot, mode: str, query: str
) -> None:
    retriever, embedder, _ = make(snapshot, mode)
    with pytest.raises(ValueError):
        retriever.retrieve(query)
    assert (embedder.query_calls, embedder.embed_calls) == (0, 0)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_empty_handle_raises_in_every_mode(mode: str) -> None:
    retriever = HybridRetriever(IndexHandle(), SpyEmbedder(), RetrievalSettings(mode=mode))  # type: ignore[arg-type]
    with pytest.raises(IndexNotReadyError):
        retriever.retrieve(QUERY)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_no_chunk_appears_twice_in_any_mode(snapshot: IndexSnapshot, mode: str) -> None:
    retriever, _, _ = make(snapshot, mode, bm25_k=20, dense_k=20)
    ids = chunk_ids(retriever.retrieve(QUERY))
    assert ids and len(ids) == len(set(ids))
