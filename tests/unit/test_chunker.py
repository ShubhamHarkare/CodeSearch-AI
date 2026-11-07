"""Tests for codesearch/ingest/chunker.py: unit cases, then every mini-corpus page."""

from pathlib import Path

import pytest

from codesearch.ingest.chunker import chunk_page, split_into_sections, split_long_section
from codesearch.ingest.mdx_clean import CleanPage, clean_mdx
from codesearch.ingest.urls import path_to_url
from codesearch.schemas import Chunk


def test_sections_and_paths() -> None:
    body = (
        "Intro paragraph.\n\n"
        "## Usage {/*usage*/}\n\nSome text.\n\n```js\nconst [n, setN] = useState(0);\n```\n\n"
        "### Adding state {/*adding-state*/}\n\nMore text.\n"
    )
    sections = split_into_sections(CleanPage(title="useState", body=body))

    assert [s.path for s in sections] == [
        ["useState"],
        ["useState", "Usage"],
        ["useState", "Usage", "Adding state"],
    ]
    assert [s.anchor for s in sections] == ["", "usage", "adding-state"]
    assert "const [n, setN] = useState(0);" in "\n".join(sections[1].lines)


def test_short_section_is_one_piece() -> None:
    assert split_long_section(["Hello.", "", "World."], max_chars=100) == ["Hello.\n\nWorld."]


def test_long_section_is_split_but_code_stays_whole() -> None:
    code = ["```js", *[f"const v{i} = {i};" for i in range(30)], "", "const end = 0;", "```"]
    lines = ["Intro text."] * 3 + [""] + code + [""] + ["Outro text."]
    pieces = split_long_section(lines, max_chars=200)

    assert len(pieces) > 1
    assert all(p.count("```") % 2 == 0 for p in pieces)  # no code block cut in half
    assert any(
        "const v0 = 0;" in p and "const end = 0;" in p for p in pieces
    )  # the big block survived whole


def test_code_fences_are_kept() -> None:
    lines = ["Intro."] * 50 + ["", "```js", "const a = 1;", "```"]
    pieces = split_long_section(lines, max_chars=100)
    assert any("```js\nconst a = 1;\n```" in p for p in pieces)


def test_repeated_blank_lines_make_no_empty_pieces() -> None:
    lines = ["", "", "First paragraph " * 5, "", "", "", "Second paragraph " * 5, "", ""]
    pieces = split_long_section(lines, max_chars=100)
    assert len(pieces) == 2
    assert all(p.strip() for p in pieces)
    assert not any("\n\n\n" in p for p in pieces)


SAMPLE = CleanPage(
    title="useState",
    body=(
        "Intro.\n\n## Usage {/*usage*/}\n\nUse it.\n\n"
        "### Adding state {/*adding-state*/}\n\nCall it.\n"
    ),
)
URL = "https://react.dev/reference/react/useState"


def test_chunk_text_starts_with_header_path() -> None:
    chunks = chunk_page(SAMPLE, URL, "reference/react/useState.md")
    assert chunks[2].text.startswith("useState > Usage > Adding state\n\n")
    assert chunks[2].anchor == "adding-state"


def test_chunk_ids_are_stable() -> None:
    first = chunk_page(SAMPLE, URL, "x.md")
    second = chunk_page(SAMPLE, URL, "x.md")
    assert [c.id for c in first] == [c.id for c in second]


def test_editing_one_section_changes_only_its_id() -> None:
    edited = CleanPage(title=SAMPLE.title, body=SAMPLE.body.replace("Call it.", "Call it now."))
    before = [c.id for c in chunk_page(SAMPLE, URL, "x.md")]
    after = [c.id for c in chunk_page(edited, URL, "x.md")]
    assert before[:2] == after[:2]  # Intro and Usage unchanged
    assert before[2] != after[2]  # only "Adding state" changed


def test_page_without_title_has_no_dangling_separator() -> None:
    chunks = chunk_page(CleanPage(title="", body="## Usage {/*usage*/}\n\nText.\n"), URL, "x.md")
    assert chunks[0].text.startswith("Usage\n\n")


# =====================================================================
# Every real page in the mini corpus
# =====================================================================

MINI_CORPUS = Path(__file__).parents[1] / "fixtures" / "mini_corpus"
PAGES = sorted(
    p for p in MINI_CORPUS.rglob("*.md") if p.name not in {"README.md", "LICENSE-DOCS.md"}
)
MAX_CHARS = 1000


def _chunks_for(page_path: Path) -> list[Chunk]:
    page = clean_mdx(page_path.read_text(encoding="utf-8"))
    return chunk_page(page, path_to_url(page_path, MINI_CORPUS), page_path.name, MAX_CHARS)


@pytest.mark.parametrize("page_path", PAGES, ids=lambda p: p.relative_to(MINI_CORPUS).as_posix())
def test_real_page_chunks(page_path: Path) -> None:
    chunks = _chunks_for(page_path)
    assert chunks, "page produced no chunks"

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk ids on one page"

    for c in chunks:
        prefix = " > ".join(c.header_path)
        assert c.text.startswith(prefix + "\n\n")
        assert c.text[len(prefix) :].strip(), "empty chunk"
        # The bug v1's splitters had: no code block may be cut in half.
        assert c.text.count("```") % 2 == 0, f"code block split in {c.header_path}"
        # Pieces respect max_chars, unless the piece is one oversized code block.
        piece = c.text[len(prefix) + 2 :]
        assert len(piece) <= MAX_CHARS or piece.count("```") >= 2, c.header_path
        # Below the page's first heading every chunk has an anchor to link to.
        if len(c.header_path) > 1:
            assert c.anchor, c.header_path


def test_use_state_chunks_link_to_real_anchors() -> None:
    chunks = _chunks_for(MINI_CORPUS / "reference/react/useState.md")
    by_anchor = {c.anchor: c for c in chunks}
    assert "usestate" in by_anchor
    assert by_anchor["usestate"].header_path == ("useState", "Reference", "useState(initialState)")
    assert "adding-state-to-a-component" in by_anchor


def test_sibling_headings_replace_each_other() -> None:
    """Regression: a second ## must replace the first, even with no # heading on the page."""
    body = (
        "## Reference {/*reference*/}\n\nA.\n\n### `memo(Component)` {/*memo*/}\n\nB.\n\n"
        "## Usage {/*usage*/}\n\nC.\n\n### Skipping re-rendering {/*skipping*/}\n\nD.\n"
    )
    paths = [s.path for s in split_into_sections(CleanPage(title="memo", body=body))]
    assert paths == [
        ["memo", "Reference"],
        ["memo", "Reference", "memo(Component)"],
        ["memo", "Usage"],
        ["memo", "Usage", "Skipping re-rendering"],
    ]


def test_long_list_without_blank_lines_is_split_at_line_breaks() -> None:
    lines = [f"{i}. " + "word " * 30 for i in range(10)]  # ~1,550 chars, no blank lines
    pieces = split_long_section(lines, max_chars=400)
    assert len(pieces) > 1
    assert all(len(p) <= 400 for p in pieces)
    assert "\n".join(pieces).count("word") == 300  # nothing lost


def test_repeated_identical_examples_get_distinct_ids() -> None:
    """Regression: useMemo.md repeats the same App.js twice inside one section."""
    example = "```js\n" + "const a = 1;\n" * 60 + "```"
    page = CleanPage(title="t", body=f"## S {{/*s*/}}\n\n{example}\n\n{example}\n")
    chunks = chunk_page(page, URL, "x.md", max_chars=500)
    assert len(chunks) == 2
    assert chunks[0].text == chunks[1].text
    assert chunks[0].id != chunks[1].id
