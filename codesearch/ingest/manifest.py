"""Records exactly what an index was built from, so results are reproducible.

What goes here:
    - ``Manifest`` (pydantic): corpus_sha, embedding_model, embedding_dim,
      chunk_params (max_chars, overlap), chunk_count, built_at, index_version,
      changelog (list of {at, added, changed, removed}).
    - ``compute_index_version(corpus_sha, embedding_model, chunk_params) -> str``:
      short hash; any change to inputs gives a new version (and so new cache keys).
    - ``read_manifest(index_dir) -> Manifest | None`` and
      ``write_manifest(index_dir, manifest)`` (write to temp file, then rename).

Tested by: tests/unit/test_manifest.py (version changes when any input changes;
round-trip read/write).
Phase: 1
"""
