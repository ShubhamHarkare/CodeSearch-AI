"""Tests for codesearch/index/handle.py."""

import dataclasses
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codesearch.errors import IndexNotReadyError
from codesearch.index.bm25 import BM25Index
from codesearch.index.embedder import HashEmbedder
from codesearch.index.handle import BM25_FILE, CHROMA_DIR, IndexHandle, IndexSnapshot
from codesearch.index.vector_store import VectorStore
from codesearch.ingest.manifest import (
    PIPELINE_VERSION,
    ChunkParams,
    Manifest,
    compute_index_version,
    write_manifest,
)
from codesearch.schemas import Chunk

EMBEDDER = HashEmbedder(dim=32)
PARAMS = ChunkParams(max_chars=1000, overlap=0)


def make_chunks(tag: str, n: int) -> list[Chunk]:
    return [
        Chunk(
            id=f"{tag}-{i}",
            url=f"https://react.dev/{tag}",
            anchor=f"a{i}",
            title=tag,
            header_path=(tag,),
            text=f"{tag} chunk number {i} about useState",
            source_path=f"{tag}.md",
            start_line=0,
        )
        for i in range(n)
    ]


def build_on_disk(
    index_dir: Path, tag: str, n: int, *, manifest_count: int | None = None
) -> IndexSnapshot:
    """Write a tiny real index (Chroma + BM25 + manifest) and open it."""
    chunks = make_chunks(tag, n)
    VectorStore(index_dir / CHROMA_DIR).upsert(chunks, EMBEDDER.embed([c.text for c in chunks]))
    BM25Index.build(chunks).save(index_dir / BM25_FILE)
    sha = (tag * 40)[:40]
    write_manifest(
        index_dir,
        Manifest(
            index_version=compute_index_version(sha, EMBEDDER.model_name, PARAMS),
            corpus_sha=sha,
            embedding_model=EMBEDDER.model_name,
            embedding_dim=EMBEDDER.dim,
            chunk_params=PARAMS,
            pipeline_version=PIPELINE_VERSION,
            chunk_count=n if manifest_count is None else manifest_count,
            built_at=datetime.now(UTC),
        ),
    )
    snapshot = IndexSnapshot.open(index_dir)
    assert snapshot is not None
    return snapshot


# =====================================================================
# IndexSnapshot.open
# =====================================================================


def test_open_on_an_empty_folder_is_none(tmp_path: Path) -> None:
    assert IndexSnapshot.open(tmp_path) is None
    assert IndexSnapshot.open(tmp_path / "missing") is None


def test_open_a_complete_build(tmp_path: Path) -> None:
    snapshot = build_on_disk(tmp_path, "a", 3)
    assert snapshot.vector_store.count() == 3
    assert len(snapshot.bm25) == 3
    assert snapshot.manifest.chunk_count == 3


def test_open_refuses_parts_that_dont_match_the_manifest(tmp_path: Path) -> None:
    with pytest.raises(IndexNotReadyError, match="manifest says 5 chunks"):
        build_on_disk(tmp_path, "a", 3, manifest_count=5)


def test_open_with_a_manifest_but_no_bm25_file(tmp_path: Path) -> None:
    build_on_disk(tmp_path, "a", 2)
    (tmp_path / BM25_FILE).unlink()
    with pytest.raises(IndexNotReadyError, match="missing"):
        IndexSnapshot.open(tmp_path)


def test_snapshots_are_frozen(tmp_path: Path) -> None:
    snapshot = build_on_disk(tmp_path, "a", 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.bm25 = BM25Index.build([])  # type: ignore[misc]


# =====================================================================
# IndexHandle
# =====================================================================


def test_empty_handle_is_not_ready() -> None:
    handle = IndexHandle()
    assert handle.ready is False
    with pytest.raises(IndexNotReadyError):
        handle.snapshot()
    with pytest.raises(IndexNotReadyError):
        _ = handle.version


def test_handle_with_a_snapshot(tmp_path: Path) -> None:
    snapshot = build_on_disk(tmp_path, "a", 2)
    handle = IndexHandle(snapshot)
    assert handle.ready is True
    assert handle.snapshot() is snapshot
    assert handle.version == snapshot.manifest.index_version


def test_swap_returns_the_old_snapshot(tmp_path: Path) -> None:
    old = build_on_disk(tmp_path / "old", "a", 2)
    new = build_on_disk(tmp_path / "new", "b", 3)
    handle = IndexHandle(old)

    assert handle.swap(new) is old
    assert handle.snapshot() is new
    assert handle.version == new.manifest.index_version != old.manifest.index_version


def test_first_swap_into_an_empty_handle(tmp_path: Path) -> None:
    handle = IndexHandle()
    assert handle.swap(build_on_disk(tmp_path, "a", 1)) is None
    assert handle.ready


def test_a_request_keeps_its_snapshot_across_a_swap(tmp_path: Path) -> None:
    """The 'request A' case: a snapshot taken before a swap still answers from the old index."""
    old = build_on_disk(tmp_path / "old", "a", 2)
    new = build_on_disk(tmp_path / "new", "b", 3)
    handle = IndexHandle(old)

    in_flight = handle.snapshot()  # request starts
    handle.swap(new)  # rebuild finishes mid-request

    assert in_flight is old
    assert {cid for cid, _ in in_flight.bm25.search("usestate", k=10)} == {"a-0", "a-1"}
    assert handle.snapshot().manifest.chunk_count == 3


def test_concurrent_swaps_never_mix_parts(tmp_path: Path) -> None:
    """Readers must always see three parts from the same build, never a mix."""
    a = build_on_disk(tmp_path / "a", "a", 2)
    b = build_on_disk(tmp_path / "b", "b", 3)
    handle = IndexHandle(a)
    stop = threading.Event()
    problems: list[str] = []

    def writer() -> None:
        while not stop.is_set():
            handle.swap(b)
            handle.swap(a)

    def reader() -> None:
        for _ in range(2000):
            snap = handle.snapshot()
            expected = snap.manifest.chunk_count
            if len(snap.bm25) != expected or snap.vector_store.path.parent.name not in "ab":
                problems.append(f"mixed snapshot: manifest {expected}, bm25 {len(snap.bm25)}")

    writers = [threading.Thread(target=writer) for _ in range(2)]
    readers = [threading.Thread(target=reader) for _ in range(4)]
    for t in writers + readers:
        t.start()
    for t in readers:
        t.join()
    stop.set()
    for t in writers:
        t.join()

    assert problems == []
