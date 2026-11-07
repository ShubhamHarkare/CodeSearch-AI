"""Hybrid retrieval: BM25 + dense search fused with Reciprocal Rank Fusion.

What goes here:
    - ``rrf_fuse(rankings: dict[str, list[str]], k=60) -> list[tuple[str, float]]``:
      pure function; score(d) = sum over lists of 1 / (k + rank_d). Ranks start at 1.
    - ``HybridRetriever(index_handle, embedder, settings)`` with
      ``retrieve(query) -> list[ScoredChunk]``:
        * mode "dense": dense top-k only (this + k=3 reproduces v1)
        * mode "bm25":  BM25 top-k only
        * mode "hybrid": BM25 top ``bm25_k`` + dense top ``dense_k`` -> rrf_fuse
        * fill EVERY score field on ScoredChunk (bm25_rank/score, dense_rank/
          score, rrf_score) so the explanation trace can show why each chunk ranked

What it must do:
    - Use one IndexSnapshot for the whole call (handle.snapshot()).

Tested by: tests/unit/test_rrf.py (hand-computed example, ties, item in one
list only) and tests/integration/test_pipeline.py.
Phase: 1
"""

from codesearch.config import RetrievalSettings
from codesearch.index.handle import IndexHandle, IndexSnapshot
from codesearch.interfaces import Embedder
from codesearch.schemas import Chunk, ScoredChunk


class HybridRetriever:
    """Finds the chunks for a query with dense search, BM25, or both fused by RRF."""

    def __init__(
        self, handle: IndexHandle, embedder: Embedder, settings: RetrievalSettings
    ) -> None:
        self.handle = handle  # gives the current index; read it once per call
        self.embedder = embedder  # turns the query into a vector (dense and hybrid only)
        self.settings = settings  # mode, bm25_k, dense_k, rrf_k

    def retrieve(self, query: str) -> list[ScoredChunk]:
        """Top chunks for ``query`` using the mode in settings (dense | bm25 | hybrid)."""
        if not query.strip():
            raise ValueError(f"query must not be blank, got {query!r}")

        # One snapshot for the whole call, so a re-index can't swap the index mid-request.
        snapshot = self.handle.snapshot()

        if self.settings.mode == "dense":
            return self._dense(snapshot, query)
        if self.settings.mode == "bm25":
            return self._bm25(snapshot, query)
        return self._hybrid(snapshot, query)

    # ------------------------------------------------------------------
    # One helper per mode. Each takes the snapshot (the index) and returns
    # ScoredChunks with only the score fields that mode produces.
    # ------------------------------------------------------------------

    def _dense(self, snapshot: IndexSnapshot, query: str) -> list[ScoredChunk]:
        """Embedding search: the dense_k chunks whose vectors are closest to the query."""
        hits = self._dense_hits(snapshot, query)
        return [
            ScoredChunk(chunk=chunk, dense_rank=rank, dense_score=score)
            for rank, (chunk, score) in enumerate(hits, start=1)
        ]

    def _bm25(self, snapshot: IndexSnapshot, query: str) -> list[ScoredChunk]:
        """Keyword search: the bm25_k chunks that best match the query's words."""
        hits = snapshot.bm25.search(query, k=self.settings.bm25_k)  # [(chunk_id, score)]
        chunks = self._chunks_by_id(snapshot, [chunk_id for chunk_id, _ in hits])
        return [
            ScoredChunk(chunk=chunks[chunk_id], bm25_rank=rank, bm25_score=score)
            for rank, (chunk_id, score) in enumerate(hits, start=1)
            if chunk_id in chunks
        ]

    def _hybrid(self, snapshot: IndexSnapshot, query: str) -> list[ScoredChunk]:
        """BM25 and dense results fused with RRF; every chunk keeps the scores it earned."""
        bm25_hits = snapshot.bm25.search(query, k=self.settings.bm25_k)
        dense_hits = self._dense_hits(snapshot, query)

        # chunk_id -> (rank, score) in each list; ranks start at 1
        bm25 = {cid: (rank, score) for rank, (cid, score) in enumerate(bm25_hits, start=1)}
        dense = {c.id: (rank, score) for rank, (c, score) in enumerate(dense_hits, start=1)}

        # Dense hits already carry their chunks; fetch only the BM25-only ones.
        chunks = {chunk.id: chunk for chunk, _ in dense_hits}
        chunks.update(self._chunks_by_id(snapshot, [cid for cid in bm25 if cid not in chunks]))

        fused = rrf_fuse({"bm25": list(bm25), "dense": list(dense)}, k=self.settings.rrf_k)

        results: list[ScoredChunk] = []
        for chunk_id, rrf_score in fused:
            if chunk_id not in chunks:
                continue
            bm25_rank, bm25_score = bm25.get(chunk_id, (None, None))
            dense_rank, dense_score = dense.get(chunk_id, (None, None))
            results.append(
                ScoredChunk(
                    chunk=chunks[chunk_id],
                    bm25_rank=bm25_rank,
                    bm25_score=bm25_score,
                    dense_rank=dense_rank,
                    dense_score=dense_score,
                    rrf_score=rrf_score,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Small shared pieces
    # ------------------------------------------------------------------

    def _dense_hits(self, snapshot: IndexSnapshot, query: str) -> list[tuple[Chunk, float]]:
        vector = self.embedder.embed_query(query)  # embed_query adds bge's query instruction
        return snapshot.vector_store.query(vector, self.settings.dense_k)

    @staticmethod
    def _chunks_by_id(snapshot: IndexSnapshot, chunk_ids: list[str]) -> dict[str, Chunk]:
        """BM25 stores only ids; the full chunks live in the vector store."""
        return {chunk.id: chunk for chunk in snapshot.vector_store.get(chunk_ids)}


def rrf_fuse(rankings: dict[str, list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum of 1 / (k + rank) over the lists holding d.

    Ranks start at 1. Best score first; ties keep first-seen order.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    ranking_per_id: dict[str, float] = {}

    for ranking in rankings.values():
        # dict.fromkeys drops repeats in one list so they neither score twice nor shift ranks
        for rank, doc_id in enumerate(dict.fromkeys(ranking), start=1):
            ranking_per_id[doc_id] = ranking_per_id.get(doc_id, 0.0) + 1.0 / (k + rank)

    # dicts keep first-insertion order and sorted() is stable, so ties stay first-seen
    return sorted(ranking_per_id.items(), key=lambda item: item[1], reverse=True)
