"""Tests for the v1-style chunker (the baseline) and the chunk_strategy setting.

v1 cut the docs into fixed 1,000-character pieces with no headings and no
anchors. We reproduce that on purpose so the eval can measure what v2's
heading-aware chunking is worth. These tests pin the "dumb" behaviour down.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from codesearch.config import IndexSettings, Settings
from codesearch.ingest.chunker import chunk_page, chunk_page_fixed
from codesearch.ingest.manifest import ChunkParams, compute_index_version
from codesearch.ingest.mdx_clean import CleanPage, clean_mdx

URL = "https://react.dev/reference/react/useState"
SRC = "reference/react/useState.md"


def page(body: str, title: str = "useState") -> CleanPage:
    return CleanPage(title=title, body=body)


def mini_page(mini_corpus_path: Path, rel: str) -> CleanPage:
    return clean_mdx((mini_corpus_path / rel).read_text(encoding="utf-8"))


# =====================================================================
# Cutting
# =====================================================================


def test_cuts_into_fixed_size_pieces() -> None:
    chunks = chunk_page_fixed(page("a" * 2500), URL, SRC, max_chars=1000)
    assert [len(c.text) for c in chunks] == [1000, 1000, 500]


def test_body_exactly_max_chars_is_one_chunk() -> None:
    assert len(chunk_page_fixed(page("x" * 1000), URL, SRC, max_chars=1000)) == 1


def test_one_char_over_max_makes_a_second_chunk() -> None:
    chunks = chunk_page_fixed(page("x" * 1001), URL, SRC, max_chars=1000)
    assert [c.text for c in chunks] == ["x" * 1000, "x"]


def test_cuts_ignore_words_and_lines() -> None:
    """v1 split blindly: a cut can land in the middle of a word."""
    chunks = chunk_page_fixed(page("hello world"), URL, SRC, max_chars=4)
    assert [c.text for c in chunks] == ["hell", "o wo", "rld"]


def test_pieces_join_back_into_the_body(mini_corpus_path: Path) -> None:
    body = mini_page(mini_corpus_path, SRC).body
    chunks = chunk_page_fixed(page(body), URL, SRC, max_chars=1000)
    assert "".join(c.text for c in chunks) == body


def test_whitespace_only_pieces_are_dropped() -> None:
    chunks = chunk_page_fixed(page("abcd" + " " * 4 + "efgh"), URL, SRC, max_chars=4)
    assert [c.text for c in chunks] == ["abcd", "efgh"]


@pytest.mark.parametrize("body", ["", "   ", "\n\n\t"])
def test_empty_body_gives_no_chunks(body: str) -> None:
    assert chunk_page_fixed(page(body), URL, SRC) == []


@pytest.mark.parametrize("max_chars", [0, -1])
def test_max_chars_must_be_positive(max_chars: int) -> None:
    with pytest.raises(ValueError):
        chunk_page_fixed(page("abc"), URL, SRC, max_chars=max_chars)


def test_default_max_chars_is_1000() -> None:
    chunks = chunk_page_fixed(page("y" * 1500), URL, SRC)
    assert [len(c.text) for c in chunks] == [1000, 500]


def test_it_can_break_code_fences_like_v1_did(mini_corpus_path: Path) -> None:
    """The point of the baseline: blind cuts split code blocks (v2 never does)."""
    p = mini_page(mini_corpus_path, SRC)
    v1 = chunk_page_fixed(p, URL, SRC, max_chars=1000)
    v2 = chunk_page(p, URL, SRC, max_chars=1000)
    broken_v1 = sum(c.text.count("```") % 2 for c in v1)
    broken_v2 = sum(c.text.count("```") % 2 for c in v2)
    assert broken_v1 > 0
    assert broken_v2 == 0


# =====================================================================
# Chunk fields
# =====================================================================


def test_chunks_have_no_anchor_and_no_header_path(mini_corpus_path: Path) -> None:
    chunks = chunk_page_fixed(mini_page(mini_corpus_path, SRC), URL, SRC)
    assert chunks
    for c in chunks:
        assert c.anchor == ""
        assert c.header_path == ()
        assert c.url == URL
        assert c.source_path == SRC
        assert c.title == "useState"


def test_no_header_prefix_is_added() -> None:
    """v2 starts each chunk with 'Title > Section'; v1 must not."""
    chunks = chunk_page_fixed(page("plain body text"), URL, SRC)
    assert [c.text for c in chunks] == ["plain body text"]


def test_ids_are_unique_within_a_page() -> None:
    chunks = chunk_page_fixed(page("same" * 300), URL, SRC, max_chars=4)
    assert len({c.id for c in chunks}) == len(chunks)


def test_ids_are_stable_across_runs(mini_corpus_path: Path) -> None:
    p = mini_page(mini_corpus_path, SRC)
    first = [c.id for c in chunk_page_fixed(p, URL, SRC)]
    second = [c.id for c in chunk_page_fixed(p, URL, SRC)]
    assert first == second


def test_ids_depend_on_the_url() -> None:
    a = chunk_page_fixed(page("text"), URL, SRC)
    b = chunk_page_fixed(page("text"), "https://react.dev/learn", SRC)
    assert a[0].id != b[0].id


def test_v1_and_v2_ids_never_collide(mini_corpus_path: Path) -> None:
    p = mini_page(mini_corpus_path, SRC)
    v1 = {c.id for c in chunk_page_fixed(p, URL, SRC)}
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
