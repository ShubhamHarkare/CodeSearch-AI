"""Gets the react.dev source at a pinned commit. Replaces v1's dataScraper.py.

What goes here:
    - ``CorpusSnapshot`` (dataclass): repo_dir, commit_sha, page_paths
      (all .md/.mdx files under src/content), fetched_at.
    - ``fetch_corpus(repo_url, sha, dest) -> CorpusSnapshot``: shallow clone on
      first run, ``git fetch`` + ``git checkout <sha>`` afterwards (subprocess,
      check=True, timeout). ``sha=None`` means "latest main" for live re-indexing.
    - ``list_pages(repo_dir) -> list[Path]``: sorted, so runs are reproducible.

What it must do:
    - Record the exact commit SHA used (goes into the manifest).
    - Raise CorpusFetchError with a clear message on git failures.

Must not:
    - run anything at import time, or clean / chunk text (that is mdx_clean / chunker).

Tested by: tests/unit/test_fetch.py with a tiny local git repo made in tmp_path.
Phase: 1
"""
