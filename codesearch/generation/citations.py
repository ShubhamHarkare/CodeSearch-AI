"""Checks and resolves the [n] citations the LLM wrote. Pure.

What goes here:
    - ``extract_citations(answer: str) -> list[int]`` (handles [1], [1][2], [1, 2]).
    - ``strip_invalid(answer, n_chunks) -> tuple[str, list[int]]``: remove any
      [n] that doesn't point at a supplied chunk; return what was removed.
    - ``to_citations(answer, chunks) -> list[Citation]``: ordered, de-duplicated,
      with react.dev URL + anchor.
    - ``sentences_with_citations(answer) -> list[tuple[str, list[int]]]``:
      feeds explain/groundedness.py.

Tested by: tests/unit/test_citations.py (formats, out-of-range removed,
code blocks ignored).
Phase: 1 (basic), 2 (sentence mapping)
"""
