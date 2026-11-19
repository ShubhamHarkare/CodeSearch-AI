"""Tests for the v1 chunker (the eval baseline) and the chunk_strategy setting.

v1 (vectorstore.py on the v1 branch) used LangChain's MarkdownHeaderTextSplitter
on #/##/### with strip_headers=False, then RecursiveCharacterTextSplitter(1000,
150). chunker.py ports both, quirks included. The pinned example below was
produced by LangChain itself, so the port can't drift from what v1 really did.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
from pydantic import ValidationError

from codesearch.config import IndexSettings, Settings
from codesearch.ingest.chunker import (
    chunk_page,
    chunk_page_v1,
    recursive_split,
    split_into_sections,
)
from codesearch.ingest.manifest import ChunkParams, compute_index_version
from codesearch.ingest.mdx_clean import CleanPage, clean_mdx

URL = "https://react.dev/reference/react/useState"
SRC = "reference/react/useState.md"


def page(body: str, title: str = "useState") -> CleanPage:
    return CleanPage(title=title, body=body)


def texts(body: str, max_chars: int = 1000, overlap: int = 150) -> list[str]:
    return [c.text for c in chunk_page_v1(page(body), URL, SRC, max_chars, overlap)]


def mini_page(mini_corpus_path: Path, rel: str) -> CleanPage:
    return clean_mdx((mini_corpus_path / rel).read_text(encoding="utf-8"))


# =====================================================================
# Matches LangChain
# =====================================================================

PINNED_BODY = (
    "Intro line.\n\n## Reference {/*reference*/}\n\n### `useState(initial)` {/*usestate*/}\n\n"
    "Call useState at the top level of your component to declare a state variable that "
    "React keeps between renders.\n\n"
    "```js\nfunction Counter() {\n\tconst [count, setCount] = useState(0);\n\n"
    "  return count;\n}\n```\n\n"
    "## Usage {/*usage*/}\n\nAdding state to a component. Updating state based on the previous "
    "state. Avoiding recreating the initial state.\n"
)

# Output of LangChain's two splitters (chunk_size=120, chunk_overlap=30) on PINNED_BODY.
PINNED_LANGCHAIN = [
    "Intro line.",
    "## Reference {/*reference*/}  \n### `useState(initial)` {/*usestate*/}",
    (
        "Call useState at the top level of your component to declare a state variable that "
        "React keeps between renders.  \n```js"
    ),
    "```js\nfunction Counter() {\nconst [count, setCount] = useState(0);",
    "return count;\n}\n```",
    "## Usage {/*usage*/}",
    (
        "Adding state to a component. Updating state based on the previous state. "
        "Avoiding recreating the initial state."
    ),
]


def test_matches_langchain_on_the_pinned_example() -> None:
    assert texts(PINNED_BODY, max_chars=120, overlap=30) == PINNED_LANGCHAIN


# =====================================================================
# Section text (MarkdownHeaderTextSplitter)
# =====================================================================


def test_heading_line_is_kept_and_no_path_prefix_is_added() -> None:
    assert texts("## Usage {/*usage*/}\n\nAdd state.") == ["## Usage {/*usage*/}  \nAdd state."]


def test_paragraphs_are_joined_with_two_spaces_and_a_newline() -> None:
    assert texts("one\ntwo\n\nthree") == ["one\ntwo  \nthree"]


def test_lines_are_stripped_and_lose_tabs() -> None:
    assert texts("```js\n    indented\n\tx\ty\n```") == ["```js\nindented\nxy\n```"]


def test_heading_only_section_is_glued_onto_the_deeper_one_after_it() -> None:
    assert texts("## A {/*a*/}\n### B {/*b*/}\ntext") == ["## A {/*a*/}  \n### B {/*b*/}\ntext"]


def test_heading_only_section_stands_alone_before_a_sibling() -> None:
    assert texts("## A\n## B\ntext") == ["## A", "## B\ntext"]


def test_v2_sections_still_drop_heading_only_sections() -> None:
    sections = split_into_sections(page("## A\n### B\ntext"))
    assert [s.path for s in sections] == [["useState", "A", "B"]]


# =====================================================================
# Size split (RecursiveCharacterTextSplitter)
# =====================================================================


def test_text_without_separators_falls_back_to_characters_with_overlap() -> None:
    assert [len(p) for p in recursive_split("a" * 2500, 1000, 150)] == [1000, 1000, 800]


def test_words_are_packed_and_overlap() -> None:
    assert recursive_split("one two three four five six", 10, 4) == [
        "one two",
        "two three",
        "four five",
        "six",
    ]


def test_long_sections_are_cut_with_overlap() -> None:
    body = "\n".join(f"line {i} " + "word " * 20 for i in range(40))
    chunks = texts(body)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    for first, second in pairwise(chunks):
        assert second.split("\n")[0] in first  # the next chunk starts inside the previous one


def test_code_blocks_can_be_cut_unlike_v2(mini_corpus_path: Path) -> None:
    """The point of the baseline: v1 splits code examples; v2 never does."""
    p = mini_page(mini_corpus_path, SRC)
    broken_v1 = sum(c.text.count("```") % 2 for c in chunk_page_v1(p, URL, SRC, max_chars=300))
    broken_v2 = sum(c.text.count("```") % 2 for c in chunk_page(p, URL, SRC, max_chars=300))
    assert broken_v1 > 0
    assert broken_v2 == 0


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (-1, 0), (100, 100), (100, -1)])
def test_bad_sizes_are_rejected(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_page_v1(page("abc"), URL, SRC, max_chars=max_chars, overlap=overlap)


@pytest.mark.parametrize("body", ["", "   ", "\n\n\t"])
def test_empty_body_gives_no_chunks(body: str) -> None:
    assert chunk_page_v1(page(body), URL, SRC) == []


# =====================================================================
# Chunk fields
# =====================================================================


def test_chunks_carry_anchor_and_header_path_as_metadata(mini_corpus_path: Path) -> None:
    chunks = chunk_page_v1(mini_page(mini_corpus_path, SRC), URL, SRC)
    assert chunks
    assert any(c.anchor for c in chunks)  # section-level metrics work for v1 too
    for c in chunks:
        assert c.url == URL
        assert c.source_path == SRC
        assert c.title == "useState"
        assert c.header_path[0] == "useState"
        assert not c.text.startswith("useState > ")


def test_ids_are_unique_and_stable(mini_corpus_path: Path) -> None:
    p = mini_page(mini_corpus_path, SRC)
    first = [c.id for c in chunk_page_v1(p, URL, SRC)]
    assert len(set(first)) == len(first)
    assert first == [c.id for c in chunk_page_v1(p, URL, SRC)]


def test_ids_depend_on_the_url() -> None:
    a = chunk_page_v1(page("text"), URL, SRC)
    b = chunk_page_v1(page("text"), "https://react.dev/learn", SRC)
    assert a[0].id != b[0].id


def test_v1_and_v2_ids_never_collide(mini_corpus_path: Path) -> None:
    p = mini_page(mini_corpus_path, SRC)
    v1 = {c.id for c in chunk_page_v1(p, URL, SRC)}
    v2 = {c.id for c in chunk_page(p, URL, SRC)}
    assert v1.isdisjoint(v2)


# =====================================================================
# Strategy setting and manifest params
# =====================================================================


def test_chunk_strategy_defaults_to_v2() -> None:
    assert IndexSettings().chunk_strategy == "v2"


def test_chunk_strategy_accepts_v1() -> None:
    assert IndexSettings(chunk_strategy="v1").chunk_strategy == "v1"


@pytest.mark.parametrize("bad", ["v3", "V1", "fixed", ""])
def test_chunk_strategy_rejects_unknown_values(bad: str) -> None:
    with pytest.raises(ValidationError):
        IndexSettings(chunk_strategy=bad)  # type: ignore[arg-type]


def test_chunk_strategy_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODESEARCH_INDEX__CHUNK_STRATEGY", "v1")
    assert Settings(_env_file=None).index.chunk_strategy == "v1"  # type: ignore[call-arg]


def test_chunk_params_strategy_defaults_to_v2() -> None:
    assert ChunkParams(max_chars=1000, overlap=0).strategy == "v2"


def test_strategy_changes_the_index_version() -> None:
    v2 = compute_index_version("a" * 40, "hash", ChunkParams(max_chars=1000, overlap=0))
    v1 = compute_index_version(
        "a" * 40, "hash", ChunkParams(max_chars=1000, overlap=0, strategy="v1")
    )
    assert v1 != v2
