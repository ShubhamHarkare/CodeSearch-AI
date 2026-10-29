"""Tests for codesearch/index/vector_store.py, against a real Chroma in tmp_path."""

from pathlib import Path

import numpy as np
import pytest

from codesearch.index.embedder import HashEmbedder
from codesearch.index.vector_store import VectorStore
from codesearch.schemas import Chunk

EMBEDDER = HashEmbedder(dim=64)


def make_chunk(chunk_id: str, text: str, header_path: tuple[str, ...] = ("useState",)) -> Chunk:
    return Chunk(
        id=chunk_id,
        url="https://react.dev/reference/react/useState",
        anchor=chunk_id,
        title="useState",
        header_path=header_path,
        text=text,
        source_path="reference/react/useState.md",
        start_line=0,
    )


CHUNKS = [
    make_chunk("a", "usestate adds state to a component", ("useState", "Usage")),
    make_chunk("b", "useeffect synchronizes with an external system", ("useEffect",)),
    make_chunk("c", "memo skips re-rendering when props are unchanged", ("memo", "Usage", "Skip")),
]


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma")


@pytest.fixture
def filled(store: VectorStore) -> VectorStore:
    store.upsert(CHUNKS, EMBEDDER.embed([c.text for c in CHUNKS]))
    return store


# =====================================================================
# upsert / get
# =====================================================================


def test_round_trip_rebuilds_equal_chunks(filled: VectorStore) -> None:
    assert filled.get(["a", "b", "c"]) == CHUNKS
    assert filled.get(["c"])[0].header_path == ("memo", "Usage", "Skip")


def test_get_keeps_requested_order_and_skips_unknown_ids(filled: VectorStore) -> None:
    assert [c.id for c in filled.get(["c", "nope", "a"])] == ["c", "a"]


def test_get_with_duplicate_or_no_ids(filled: VectorStore) -> None:
    assert [c.id for c in filled.get(["a", "a"])] == ["a", "a"]
    assert filled.get([]) == []


def test_upsert_is_idempotent(filled: VectorStore) -> None:
    filled.upsert(CHUNKS, EMBEDDER.embed([c.text for c in CHUNKS]))
    assert filled.count() == 3


def test_upsert_updates_text_in_place(filled: VectorStore) -> None:
    changed = make_chunk("a", "brand new text")
    filled.upsert([changed], EMBEDDER.embed([changed.text]))
    assert filled.count() == 3
    assert filled.get(["a"])[0].text == "brand new text"


def test_empty_header_path_and_empty_anchor_survive(store: VectorStore) -> None:
    """Chroma rejects empty lists as metadata; header_path is stored as JSON."""
    chunk = make_chunk("x", "intro text", header_path=())
    chunk = chunk.model_copy(update={"anchor": ""})
    store.upsert([chunk], EMBEDDER.embed([chunk.text]))
    assert store.get(["x"]) == [chunk]


def test_unicode_and_special_characters_survive(store: VectorStore) -> None:
    chunk = make_chunk("u", 'Les "guillemets" — 日本語 — `code` > path', ("a > b", "c"))
    store.upsert([chunk], EMBEDDER.embed([chunk.text]))
    assert store.get(["u"]) == [chunk]


def test_length_mismatch_raises(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="3 chunks but 2 vectors"):
        store.upsert(CHUNKS, EMBEDDER.embed(["one", "two"]))


def test_duplicate_ids_in_one_upsert_raise(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        store.upsert([CHUNKS[0], CHUNKS[0]], EMBEDDER.embed(["x", "x"]))


def test_upserting_nothing_is_a_no_op(store: VectorStore) -> None:
    store.upsert([], np.zeros((0, 64), dtype=np.float32))
    assert store.count() == 0


def test_upsert_in_several_batches(store: VectorStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store.client, "get_max_batch_size", lambda: 2)
    chunks = [make_chunk(f"id{i}", f"text number {i}") for i in range(5)]
    store.upsert(chunks, EMBEDDER.embed([c.text for c in chunks]))
    assert store.count() == 5
    assert store.all_ids() == {f"id{i}" for i in range(5)}


# =====================================================================
# query
# =====================================================================


def test_query_with_a_chunks_own_vector_finds_it_first(filled: VectorStore) -> None:
    results = filled.query(EMBEDDER.embed_query(CHUNKS[1].text), k=3)
    best, score = results[0]
    assert best.id == "b"
    assert score == pytest.approx(1.0, abs=1e-4)


def test_query_scores_are_similarities_sorted_best_first(filled: VectorStore) -> None:
    scores = [score for _, score in filled.query(EMBEDDER.embed_query("usestate state"), k=3)]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 - 1e-6 <= s <= 1.0 + 1e-6 for s in scores)


def test_query_returns_real_chunks(filled: VectorStore) -> None:
    chunk, _ = filled.query(EMBEDDER.embed_query("memo props"), k=1)[0]
    assert isinstance(chunk, Chunk)
    assert chunk in CHUNKS


def test_query_on_empty_store_is_empty(store: VectorStore) -> None:
    assert store.query(EMBEDDER.embed_query("anything"), k=5) == []


def test_k_larger_than_store_returns_everything(filled: VectorStore) -> None:
    assert len(filled.query(EMBEDDER.embed_query("x"), k=50)) == 3


def test_k_zero_returns_nothing(filled: VectorStore) -> None:
    assert filled.query(EMBEDDER.embed_query("x"), k=0) == []


# =====================================================================
# delete / all_ids / count / persistence
# =====================================================================


def test_delete_removes_only_given_ids(filled: VectorStore) -> None:
    filled.delete(["a", "c"])
    assert filled.all_ids() == {"b"}
    assert filled.count() == 1


def test_delete_nothing_or_unknown_ids_is_harmless(filled: VectorStore) -> None:
    filled.delete([])
    filled.delete(["does-not-exist"])
    assert filled.count() == 3


def test_all_ids_matches_what_was_upserted(filled: VectorStore) -> None:
    assert filled.all_ids() == {"a", "b", "c"}


def test_data_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    VectorStore(path).upsert(CHUNKS, EMBEDDER.embed([c.text for c in CHUNKS]))

    reopened = VectorStore(path)  # a fresh object, as the API would create at startup
    assert reopened.count() == 3
    assert reopened.get(["b"]) == [CHUNKS[1]]


def test_collections_are_separate(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    VectorStore(path, "docs_one").upsert(CHUNKS[:1], EMBEDDER.embed([CHUNKS[0].text]))
    assert VectorStore(path, "docs_two").count() == 0
