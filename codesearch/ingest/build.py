"""Runs ingestion end to end, incrementally. Used by the CLI and live re-indexing.

What goes here:
    - ``IngestReport`` (dataclass): corpus_sha, added, changed, removed,
      unchanged, duration_s, index_version.
    - ``build_index(settings, embedder, handle: IndexHandle | None, full=False) -> IngestReport``:
        1. fetch_corpus at the configured SHA (or latest for live updates)
        2. clean_mdx + path_to_url + chunk_page for every page
        3. diff new chunk ids against the live vector store's ids
        4. embed ONLY new chunks (batched), upsert them, delete removed ids
        5. rebuild BM25 over all chunks
        6. write the manifest (with a changelog entry)
        7. ``handle.swap(...)`` so queries move to the new index atomically

What it must do:
    - Be safe to re-run: a second run with no upstream changes embeds nothing.

Must not:
    - be callable twice at once (api/background.py holds the single-flight lock).

Tested by: tests/integration/test_reindex.py (edit one fixture page ->
exactly that page's chunks change; new text becomes retrievable).
Phase: 1 (full build), 3 (incremental + swap)
"""
