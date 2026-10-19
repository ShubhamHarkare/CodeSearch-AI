"""Holds the live index and swaps in a new one without downtime.

What goes here:
    - ``IndexSnapshot`` (frozen dataclass): vector_store, bm25, manifest.
    - ``IndexHandle``:
        * ``snapshot() -> IndexSnapshot``: what a request uses for its whole
          lifetime (read the reference once, no lock needed)
        * ``swap(new: IndexSnapshot) -> None``: replace the reference under a
          ``threading.Lock``; log old and new index_version
        * ``version -> str``: current index_version (part of every cache key)
        * ``ready -> bool``: for /readyz

What it must do:
    - Guarantee a request never sees a half-built index: build the new BM25
      and manifest fully BEFORE swap().

Tested by: tests/unit/test_handle.py (snapshot taken before swap keeps the old
version; version changes after swap).
Phase: 1 (simple holder), 3 (swap for live re-indexing)
"""
