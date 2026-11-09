"""Tests for codesearch/generation/citations.py: reading the [n] citations an LLM wrote.

The answer prompt numbers the context chunks [1]..[n] and asks the model to cite them.
Small local models make mistakes: citing [7] when only 5 chunks were sent, writing
[1, 2] instead of [1][2], or putting array indexes like items[0] in code. These
functions find the real citations, drop the impossible ones, and map the rest back
to their chunks.
"""

from __future__ import annotations

import pytest

from codesearch.generation.citations import extract_citations, strip_invalid, to_citations
from codesearch.schemas import Chunk, Citation, ScoredChunk


def sc(n: int, anchor: str = "") -> ScoredChunk:
    chunk = Chunk(
        id=f"chunk-{n}",
        url=f"https://react.dev/page{n}",
        anchor=anchor,
        title=f"Page {n}",
        header_path=(f"Page {n}",),
        text="x",
        source_path=f"page{n}.md",
        start_line=0,
    )
    return ScoredChunk(chunk=chunk, dense_rank=n, dense_score=0.8)


CHUNKS = [sc(1, "usestate"), sc(2), sc(3, "caveats")]

# =====================================================================
# extract_citations
# =====================================================================


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("useState adds state [1].", [1]),
        ("Two sources [1][2].", [1, 2]),
        ("Two sources [1] [2].", [1, 2]),
        ("Comma style [1, 2].", [1, 2]),
        ("No space [1,3].", [1, 3]),
        ("Big number [12].", [12]),
        ("No citations at all.", []),
        ("", []),
    ],
)
def test_finds_citation_numbers(answer: str, expected: list[int]) -> None:
    assert extract_citations(answer) == expected


def test_order_of_first_appearance_without_duplicates() -> None:
    answer = "First [2]. Then [1]. Again [2]. And [3][1]."
    assert extract_citations(answer) == [2, 1, 3]


def test_ignores_fenced_code_blocks() -> None:
    answer = "Read the first item [1].\n\n```js\nconst first = items[0];\nlist[2] = x;\n```\n"
    assert extract_citations(answer) == [1]


def test_ignores_inline_code() -> None:
    assert extract_citations("Use `arr[1]` to read it [2].") == [2]


def test_ignores_brackets_that_are_not_just_numbers() -> None:
    answer = "See [the docs](https://react.dev) and [note] and [1a] and [-1] but [3]."
    assert extract_citations(answer) == [3]


def test_zero_is_returned_so_strip_invalid_can_remove_it() -> None:
    """extract reports what was written; validity is strip_invalid's job."""
    assert extract_citations("Odd [0].") == [0]


# =====================================================================
# strip_invalid
# =====================================================================


def test_keeps_an_answer_with_only_valid_citations_unchanged() -> None:
    answer = "useState adds state [1]. It returns a pair [2][3]."
    assert strip_invalid(answer, n_chunks=3) == (answer, [])


def test_removes_a_number_beyond_the_chunks_sent() -> None:
    cleaned, removed = strip_invalid("Real [1]. Invented [7].", n_chunks=3)
    assert cleaned == "Real [1]. Invented."
    assert removed == [7]


def test_removes_zero() -> None:
    cleaned, removed = strip_invalid("Zero [0].", n_chunks=3)
    assert (cleaned, removed) == ("Zero.", [0])


def test_keeps_the_valid_numbers_inside_a_group() -> None:
    cleaned, removed = strip_invalid("Mixed [1, 9, 2].", n_chunks=3)
    assert (cleaned, removed) == ("Mixed [1, 2].", [9])


def test_removes_a_whole_invalid_group_and_its_leading_space() -> None:
    cleaned, _ = strip_invalid("Claim [8, 9]. Next.", n_chunks=3)
    assert cleaned == "Claim. Next."


def test_removed_numbers_are_unique_and_in_order() -> None:
    _, removed = strip_invalid("A [9]. B [5]. C [9].", n_chunks=3)
    assert removed == [9, 5]


def test_with_no_chunks_every_citation_is_removed() -> None:
    cleaned, removed = strip_invalid("Claim [1].", n_chunks=0)
    assert (cleaned, removed) == ("Claim.", [1])


def test_code_blocks_are_never_changed() -> None:
    answer = "Index it [1].\n\n```js\nconst x = items[9];\n```\nInline `list[7]` too."
    cleaned, removed = strip_invalid(answer, n_chunks=1)
    assert cleaned == answer
    assert removed == []


def test_negative_chunk_count_is_an_error() -> None:
    with pytest.raises(ValueError):
        strip_invalid("x", n_chunks=-1)


# =====================================================================
# to_citations
# =====================================================================


def test_maps_numbers_to_their_chunks() -> None:
    citations = to_citations("A [1]. B [3].", CHUNKS)
    assert citations == [
        Citation(
            number=1, chunk_id="chunk-1", url="https://react.dev/page1#usestate", title="Page 1"
        ),
        Citation(
            number=3, chunk_id="chunk-3", url="https://react.dev/page3#caveats", title="Page 3"
        ),
    ]


def test_number_n_means_the_nth_chunk_sent() -> None:
    """[2] is chunks[1]: the position in the list, matching format_context's numbering."""
    (citation,) = to_citations("B [2].", CHUNKS)
    assert citation.chunk_id == "chunk-2"


def test_url_has_no_hash_for_an_anchorless_chunk() -> None:
    (citation,) = to_citations("B [2].", CHUNKS)
    assert citation.url == "https://react.dev/page2"


def test_each_number_appears_once_in_first_appearance_order() -> None:
    citations = to_citations("X [3]. Y [1]. Z [3][1].", CHUNKS)
    assert [c.number for c in citations] == [3, 1]


def test_out_of_range_numbers_are_skipped() -> None:
    citations = to_citations("Real [1]. Invented [7]. Zero [0].", CHUNKS)
    assert [c.number for c in citations] == [1]


def test_no_citations_gives_an_empty_list() -> None:
    assert to_citations("Nothing cited here.", CHUNKS) == []


def test_numbers_inside_code_are_not_citations() -> None:
    assert to_citations("Code `items[2]` only.", CHUNKS) == []
