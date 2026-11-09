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

import re
from collections.abc import Iterator

from codesearch.ingest.urls import url_with_anchor
from codesearch.schemas import Citation, ScoredChunk

_CODE = re.compile(r"```.*?(?:```|\Z)|`[^`\n]*`", re.S)  # fenced or inline code
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [1], [1, 2], [1,2]
# Same as _CITE, plus the whitespace before the bracket (group 1), so removing a
# citation doesn't leave "Invented ." behind.
_CITE_WITH_SPACE = re.compile(r"(\s*)\[(\d+(?:\s*,\s*\d+)*)\]")


def _segments(text: str) -> Iterator[tuple[str, bool]]:
    """Split text into (piece, is_code) parts, so code is never read or changed."""
    position = 0
    for match in _CODE.finditer(text):
        yield text[position : match.start()], False  # prose before this code
        yield match.group(0), True  # the code itself
        position = match.end()
    yield text[position:], False  # prose after the last code block


def extract_citations(answer: str) -> list[int]:
    """Every cited number, in order of first appearance, without duplicates (code ignored)."""
    citations: list[int] = []
    for piece, is_code in _segments(answer):
        if is_code:
            continue
        for match in _CITE.finditer(piece):
            for part in match.group(1).split(","):  # "1, 3" -> ["1", " 3"]
                number = int(part)  # int() ignores the spaces
                if number not in citations:
                    citations.append(number)
    return citations


def strip_invalid(answer: str, n_chunks: int) -> tuple[str, list[int]]:
    """Remove citations that point at no chunk sent (outside 1..n_chunks).

    Returns the cleaned answer and the removed numbers (unique, first-seen order).
    Code is never changed.
    """
    if n_chunks < 0:
        raise ValueError(f"n_chunks must be >= 0, got {n_chunks}")
    removed: list[int] = []

    def fix(match: re.Match[str]) -> str:
        """Called by re.sub for each [..]; what it returns replaces the match."""
        space, inside = match.group(1), match.group(2)  # " ", "1, 9, 2"
        numbers = [int(part) for part in inside.split(",")]  # [1, 9, 2]
        keep = [n for n in numbers if 1 <= n <= n_chunks]  # [1, 2]

        for n in numbers:
            if not 1 <= n <= n_chunks and n not in removed:
                removed.append(n)

        if not keep:  # nothing valid: drop the bracket and the space before it
            return ""
        if keep == numbers:  # all valid: leave it exactly as written
            return match.group(0)
        return space + "[" + ", ".join(str(n) for n in keep) + "]"  # " [1, 2]"

    parts: list[str] = []
    for piece, is_code in _segments(answer):
        parts.append(piece if is_code else _CITE_WITH_SPACE.sub(fix, piece))
    return "".join(parts), removed


def to_citations(answer: str, chunks: list[ScoredChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for number in extract_citations(answer):  # already unique, in order
        if 1 <= number <= len(chunks):  # skip [0] and invented numbers
            chunk = chunks[number - 1].chunk  # [1] is chunks[0]
            citations.append(
                Citation(
                    number=number,
                    chunk_id=chunk.id,
                    url=url_with_anchor(chunk.url, chunk.anchor),
                    title=chunk.title,
                )
            )
    return citations
