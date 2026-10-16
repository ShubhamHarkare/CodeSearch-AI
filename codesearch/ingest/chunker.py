"""Splits a cleaned page into heading-aware chunks with stable IDs. Pure.

What goes here:
    - ``chunk_page(page: CleanPage, url: str, max_chars: int, overlap: int) -> list[Chunk]``:
        * split on #, ##, ### headings, tracking the header path
          (e.g. ["useState", "Troubleshooting", "I've updated the state..."])
        * prepend the header path to each chunk's text ("useState > Troubleshooting > ...")
          so the embedder and BM25 both see it
        * if a section is longer than max_chars, split on paragraph boundaries,
          NEVER inside a ``` code fence (a single huge fence stays whole)
        * set anchor from the nearest heading (urls.heading_anchor)
    - ``chunk_id(url, header_path, text) -> str``: sha256 hex of the three, so
      the same content always gets the same id (enables incremental re-indexing).

Must not:
    - embed, store, or read files.

Tested by: tests/unit/test_chunker.py (header path present, code fence not
split, ids stable across runs, ids change when text changes, max_chars respected).
Phase: 1
"""
