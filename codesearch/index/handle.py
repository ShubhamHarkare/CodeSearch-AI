"""Holds the live index and swaps in a new one without downtime.

The index is three parts that must always come from the same build:
``VectorStore`` (index/chroma/), ``BM25Index`` (index/bm25.json) and the
``Manifest`` (index/manifest.json). ``IndexSnapshot`` keeps them together, and
``IndexHandle`` is the single place the app asks "what is the current index?".

Rule for callers: take ``handle.snapshot()`` ONCE per request and use that
snapshot for the whole request. A rebuild can ``swap`` in a new snapshot at any
moment; a request that already holds the old one keeps using it, so no request
ever mixes vectors from one build with BM25 from another.

Why no lock in ``snapshot()``: reading or replacing one attribute is atomic in
Python, so readers always see a whole old or a whole new snapshot. The lock in
``swap()`` only stops two writers from interleaving.

Tested by: tests/unit/test_handle.py.
Phase: 1 (holder), 3 (live swap during background re-indexing)
"""

import threading
from dataclasses import dataclass
from pathlib import Path

from codesearch.errors import IndexNotReadyError
from codesearch.index.bm25 import BM25Index
from codesearch.index.vector_store import VectorStore
from codesearch.ingest.manifest import Manifest, read_manifest
from codesearch.log import get_logger

log = get_logger(__name__)

CHROMA_DIR = "chroma"
BM25_FILE = "bm25.json"


@dataclass(frozen=True)
class IndexSnapshot:
    """The three parts of one build. Frozen: a snapshot never changes after creation."""

    vector_store: VectorStore
    bm25: BM25Index
    manifest: Manifest

    @classmethod
    def open(cls, index_dir: Path) -> "IndexSnapshot | None":
        """Open a built index from disk; None if nothing has been built there yet."""
        index_dir = Path(index_dir)
        manifest = read_manifest(index_dir)
        if manifest is None:
            return None
        bm25_path = index_dir / BM25_FILE
        if not bm25_path.exists():
            raise IndexNotReadyError(f"{bm25_path} is missing but a manifest exists")
        bm25 = BM25Index.load(bm25_path)
        vector_store = VectorStore(index_dir / CHROMA_DIR)

        # Cheap check that the parts belong to the same, finished build.
        expected = manifest.chunk_count
        if vector_store.count() != expected or len(bm25) != expected:
            raise IndexNotReadyError(
                f"index files don't match the manifest: manifest says {expected} chunks, "
                f"vector store has {vector_store.count()}, BM25 has {len(bm25)}"
            )
        return cls(vector_store=vector_store, bm25=bm25, manifest=manifest)


class IndexHandle:
    """Holds the current snapshot; lets a new build replace it safely."""

    def __init__(self, snapshot: IndexSnapshot | None = None) -> None:
        self._current = snapshot
        self._lock = threading.Lock()

    def snapshot(self) -> IndexSnapshot:
        """The current snapshot. Call once per request and keep using the result."""
        current = self._current  # read the reference exactly once
        if current is None:
            raise IndexNotReadyError("no index has been built or loaded yet")
        return current

    def swap(self, new: IndexSnapshot) -> IndexSnapshot | None:
        """Make ``new`` the current snapshot; returns the one it replaced."""
        with self._lock:
            old = self._current
            self._current = new
        log.info(
            "index.swapped",
            old_version=old.manifest.index_version if old else None,
            new_version=new.manifest.index_version,
            chunks=new.manifest.chunk_count,
        )
        return old

    @property
    def ready(self) -> bool:
        """True once a snapshot is loaded (used by /readyz)."""
        return self._current is not None

    @property
    def version(self) -> str:
        """The current index_version (part of every cache key)."""
        return self.snapshot().manifest.index_version
