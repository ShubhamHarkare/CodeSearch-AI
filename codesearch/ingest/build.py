"""Runs ingestion end to end, incrementally. Used by ``codesearch ingest``.

Steps:
    1. ``fetch_corpus``: react.dev at the configured commit
    2. ``collect_chunks``: clean_mdx + path_to_url + chunk_page for every page
    3. compare the new chunk ids with the ids already in the vector store
    4. embed ONLY new chunks, upsert them, delete ids that disappeared
    5. rebuild BM25 over all chunks (fast; it is not incremental)
    6. write the manifest, with a changelog line for this build
    7. open the finished index and, if a handle was given, swap it in

Why "only new chunks": chunk ids are content hashes, so an unchanged chunk keeps
its id and its stored vector is still right. A second run with no upstream
changes embeds nothing.

Full rebuild (every chunk re-embedded) when asked with ``full=True``, or when
the stored vectors came from a different embedding model or dimension: a chunk
id says nothing about which model embedded it.

Must not:
    - be run twice at once on the same index_dir (api/background.py will hold a
      lock in Phase 3).

Tested by: tests/integration/test_build.py.
Phase: 1 (full + incremental build), 3 (called by live re-indexing)
"""

import time
from dataclasses import dataclass
from datetime import UTC, datetime

from codesearch.config import Settings
from codesearch.index.bm25 import BM25Index
from codesearch.index.handle import BM25_FILE, CHROMA_DIR, IndexHandle, IndexSnapshot
from codesearch.index.vector_store import VectorStore
from codesearch.ingest.chunker import chunk_page
from codesearch.ingest.fetch import CorpusSnapshot, fetch_corpus
from codesearch.ingest.manifest import (
    PIPELINE_VERSION,
    ChangeLogEntry,
    ChunkParams,
    Manifest,
    compute_index_version,
    read_manifest,
    write_manifest,
)
from codesearch.ingest.mdx_clean import clean_mdx
from codesearch.ingest.urls import path_to_url
from codesearch.interfaces import Embedder
from codesearch.log import get_logger
from codesearch.schemas import Chunk

log = get_logger(__name__)

REPO_DIR_NAME = "react.dev"
EMBED_BATCH = 256  # chunks embedded per call, so progress shows up in the logs
MAX_CHANGELOG = 50


@dataclass(frozen=True)
class IngestReport:
    """What one build did."""

    corpus_sha: str
    index_version: str
    pages: int
    chunks: int
    added: int
    removed: int
    unchanged: int
    full_rebuild: bool
    duration_s: float


def collect_chunks(corpus: CorpusSnapshot, params: ChunkParams) -> list[Chunk]:
    """Clean and chunk every page of a fetched corpus."""
    chunks: list[Chunk] = []
    for page_path in corpus.page_paths:
        page = clean_mdx(page_path.read_text(encoding="utf-8"))
        chunks.extend(
            chunk_page(
                page,
                url=path_to_url(page_path, corpus.content_root),
                source_path=page_path.relative_to(corpus.content_root).as_posix(),
                max_chars=params.max_chars,
                overlap=params.overlap,
            )
        )
    return chunks


def build_index(
    settings: Settings,
    embedder: Embedder,
    *,
    handle: IndexHandle | None = None,
    full: bool = False,
) -> IngestReport:
    """Fetch, chunk, embed what's new, and write a complete index to settings.index.index_dir."""
    started = time.perf_counter()
    cfg = settings.index
    index_dir = cfg.index_dir

    # 1-2. Fetch and chunk.
    corpus = fetch_corpus(cfg.react_dev_repo, cfg.react_dev_sha, cfg.corpus_dir / REPO_DIR_NAME)
    params = ChunkParams(max_chars=cfg.max_chars, overlap=cfg.chunk_overlap)
    chunks = collect_chunks(corpus, params)
    log.info("build.chunked", pages=len(corpus.page_paths), chunks=len(chunks))

    # Decide whether stored vectors can be reused.
    previous = read_manifest(index_dir)
    store = VectorStore(index_dir / CHROMA_DIR)
    full_rebuild = full or _vectors_incompatible(previous, embedder)
    if full_rebuild:
        store.reset()
        log.info("build.full_rebuild", requested=full, previous_model=_model_of(previous))

    # 3-4. Embed and store only what changed.
    new_ids = {chunk.id for chunk in chunks}
    stored_ids = store.all_ids()
    to_add = [chunk for chunk in chunks if chunk.id not in stored_ids]
    to_remove = sorted(stored_ids - new_ids)
    for start in range(0, len(to_add), EMBED_BATCH):
        batch = to_add[start : start + EMBED_BATCH]
        store.upsert(batch, embedder.embed([chunk.text for chunk in batch]))
        log.info("build.embedded", done=min(start + EMBED_BATCH, len(to_add)), of=len(to_add))
    store.delete(to_remove)

    # 5. BM25 over everything.
    BM25Index.build(chunks).save(index_dir / BM25_FILE)

    # 6. Manifest.
    index_version = compute_index_version(corpus.commit_sha, embedder.model_name, params)
    now = datetime.now(UTC)
    entry = ChangeLogEntry(
        at=now,
        corpus_sha=corpus.commit_sha,
        added=len(to_add),
        removed=len(to_remove),
        unchanged=len(chunks) - len(to_add),
    )
    history = [] if previous is None else previous.changelog
    write_manifest(
        index_dir,
        Manifest(
            index_version=index_version,
            corpus_sha=corpus.commit_sha,
            embedding_model=embedder.model_name,
            embedding_dim=embedder.dim,
            chunk_params=params,
            pipeline_version=PIPELINE_VERSION,
            chunk_count=len(chunks),
            built_at=now,
            changelog=[*history, entry][-MAX_CHANGELOG:],
        ),
    )

    # 7. Open what we wrote (this also checks the three parts agree) and swap it in.
    snapshot = IndexSnapshot.open(index_dir)
    if handle is not None and snapshot is not None:
        handle.swap(snapshot)

    report = IngestReport(
        corpus_sha=corpus.commit_sha,
        index_version=index_version,
        pages=len(corpus.page_paths),
        chunks=len(chunks),
        added=entry.added,
        removed=entry.removed,
        unchanged=entry.unchanged,
        full_rebuild=full_rebuild,
        duration_s=round(time.perf_counter() - started, 2),
    )
    log.info("build.done", **report.__dict__)
    return report


def _vectors_incompatible(previous: Manifest | None, embedder: Embedder) -> bool:
    """True if stored vectors were made by a different model (ids alone can't tell)."""
    if previous is None:
        return False
    return previous.embedding_model != embedder.model_name or (
        previous.embedding_dim != embedder.dim
    )


def _model_of(manifest: Manifest | None) -> str | None:
    return None if manifest is None else manifest.embedding_model
