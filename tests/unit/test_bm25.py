"""Tests for codesearch/index/bm25.py: tokenizer, search, explanations, save/load."""

import json
from pathlib import Path

import pytest

from codesearch.index.bm25 import BM25Index, tokenize
from codesearch.ingest.chunker import chunk_page
from codesearch.ingest.fetch import list_pages
from codesearch.ingest.mdx_clean import clean_mdx
from codesearch.ingest.urls import path_to_url
from codesearch.schemas import Chunk

# =====================================================================
# tokenize
# =====================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Call useLayoutEffect before paint",
            ["call", "uselayouteffect", "use", "layout", "effect", "before", "paint"],
        ),
        ("setCount(0)", ["setcount", "set", "count"]),
        ("HTMLElement", ["htmlelement", "html", "element"]),
        ("XMLHttpRequest", ["xmlhttprequest", "xml", "http", "request"]),
        ("use_state snake", ["use", "state", "snake"]),
        ("React 19.2 release", ["react", "19", "release"]),
        ("## Usage {/*usage*/} **bold** `code`", ["usage", "usage", "bold", "code"]),
    ],
    ids=["camel", "call", "acronym", "mixed-acronym", "snake", "numbers", "markdown"],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected


def test_single_characters_are_dropped() -> None:
    assert tokenize("a b c I x 1") == []


@pytest.mark.parametrize("text", ["", "   ", "!!! --- ### ```", "\n\t"])
def test_empty_or_punctuation_only_gives_no_tokens(text: str) -> None:
    assert tokenize(text) == []


def test_non_english_letters_are_kept_whole() -> None:
    assert tokenize("café naïve 日本語") == ["café", "naïve", "日本語"]


def test_repeated_words_are_kept_for_term_frequency() -> None:
    assert tokenize("useState useState").count("usestate") == 2


def test_case_is_folded() -> None:
    assert tokenize("USESTATE useState UseState")[:1] == ["usestate"]
    assert set(tokenize("ID iD Id")) == {"id"}


# =====================================================================
# Building the index
# =====================================================================


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        url=f"https://react.dev/{chunk_id}",
        anchor=chunk_id,
        title=chunk_id,
        header_path=(chunk_id,),
        text=text,
        source_path=f"{chunk_id}.md",
        start_line=0,
    )


HOOKS = [
    make_chunk("use-effect", "useEffect lets you synchronize a component with an external system"),
    make_chunk("use-layout-effect", "useLayoutEffect fires before the browser repaints the screen"),
    make_chunk("use-state", "useState lets you add a state variable to your component"),
    make_chunk("memo", "memo lets you skip re-rendering when props are unchanged"),
]


@pytest.fixture
def index() -> BM25Index:
    return BM25Index.build(HOOKS)


def test_build_keeps_ids_in_chunk_order(index: BM25Index) -> None:
    assert index.ids == [c.id for c in HOOKS]
    assert len(index) == 4
    assert index.positions["memo"] == 3


def test_mismatched_ids_and_tokens_raise() -> None:
    with pytest.raises(ValueError, match="2 ids but 1 token lists"):
        BM25Index(["a", "b"], [["word"]])


# =====================================================================
# search
# =====================================================================


def test_exact_api_name_ranks_first(index: BM25Index) -> None:
    """The reason BM25 exists next to vectors: exact names must win."""
    results = index.search("useLayoutEffect", k=4)
    assert results[0][0] == "use-layout-effect"


def test_camel_case_parts_give_partial_matches(index: BM25Index) -> None:
    ids = [cid for cid, _ in index.search("layout effect", k=4)]
    assert ids[0] == "use-layout-effect"
    assert "use-effect" in ids  # shares "effect"


def test_results_are_best_first_and_capped_at_k(index: BM25Index) -> None:
    results = index.search("lets you component state", k=2)
    assert len(results) == 2
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_only_chunks_sharing_a_word_are_returned(index: BM25Index) -> None:
    ids = {cid for cid, _ in index.search("memo props", k=10)}
    assert ids == {"memo"}


def test_no_overlap_means_no_results(index: BM25Index) -> None:
    assert index.search("kubernetes helm chart", k=5) == []


def test_negative_idf_matches_are_still_returned() -> None:
    """rank-bm25 gives words in >half of chunks a NEGATIVE score.

    Matching must be decided by word overlap, not by score > 0, or these vanish.
    """
    tiny = BM25Index(["one", "two"], [["alpha", "beta"], ["alpha", "gamma"]])
    results = tiny.search("alpha", k=5)
    assert {cid for cid, _ in results} == {"one", "two"}
    assert all(score < 0 for _, score in results)
    assert tiny.search("delta", k=5) == []


@pytest.mark.parametrize("query", ["", "   ", "?!", "a"])
def test_empty_queries_return_nothing(index: BM25Index, query: str) -> None:
    assert index.search(query, k=5) == []


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_returns_nothing(index: BM25Index, k: int) -> None:
    assert index.search("useState", k=k) == []


def test_k_larger_than_the_index_is_fine(index: BM25Index) -> None:
    assert len(index.search("lets you", k=100)) == 3


def test_empty_index_works() -> None:
    empty = BM25Index.build([])
    assert len(empty) == 0
    assert empty.search("useState", k=5) == []


def test_scores_are_plain_floats(index: BM25Index) -> None:
    _, score = index.search("useState", k=1)[0]
    assert type(score) is float


# =====================================================================
# term_contributions
# =====================================================================


def test_contributions_sum_to_the_search_score() -> None:
    idx = BM25Index(["c1", "c2", "c3"], [["use", "state", "hook"], ["use", "effect"], ["memo"]])
    score = dict(idx.search("use state", 3))["c1"]
    parts = idx.term_contributions("use state", "c1")
    assert set(parts) == {"use", "state"}
    assert sum(parts.values()) == pytest.approx(score)


def test_contributions_only_include_words_in_that_chunk() -> None:
    idx = BM25Index(["c1", "c2", "c3"], [["use", "state", "hook"], ["use", "effect"], ["memo"]])
    assert set(idx.term_contributions("use state", "c2")) == {"use"}
    assert idx.term_contributions("use state", "c3") == {}


def test_contributions_keep_query_order_without_duplicates(index: BM25Index) -> None:
    parts = index.term_contributions("state usestate state use", "use-state")
    assert list(parts) == ["state", "usestate", "use"]


def test_contributions_for_every_search_result_add_up(index: BM25Index) -> None:
    query = "useLayoutEffect component effect"
    for chunk_id, score in index.search(query, k=4):
        assert sum(index.term_contributions(query, chunk_id).values()) == pytest.approx(score)


def test_contributions_for_unknown_chunk_raise(index: BM25Index) -> None:
    with pytest.raises(KeyError):
        index.term_contributions("useState", "no-such-chunk")


def test_contributions_for_an_empty_query(index: BM25Index) -> None:
    assert index.term_contributions("", "memo") == {}


# =====================================================================
# save / load
# =====================================================================


def test_save_then_load_gives_identical_results(index: BM25Index, tmp_path: Path) -> None:
    path = tmp_path / "index" / "bm25.json"  # parent folder doesn't exist yet
    index.save(path)
    loaded = BM25Index.load(path)

    assert loaded.ids == index.ids
    for query in ["useLayoutEffect", "lets you component", "memo props"]:
        assert loaded.search(query, k=4) == index.search(query, k=4)


def test_saved_file_is_plain_json_and_no_temp_file_remains(
    index: BM25Index, tmp_path: Path
) -> None:
    path = tmp_path / "bm25.json"
    index.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"ids", "tokens"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bm25.json"]


def test_save_overwrites(index: BM25Index, tmp_path: Path) -> None:
    path = tmp_path / "bm25.json"
    BM25Index.build(HOOKS[:1]).save(path)
    index.save(path)
    assert len(BM25Index.load(path)) == 4


def test_empty_index_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "bm25.json"
    BM25Index.build([]).save(path)
    loaded = BM25Index.load(path)
    assert len(loaded) == 0
    assert loaded.search("anything", k=3) == []


def test_loading_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BM25Index.load(tmp_path / "nope.json")


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("not json", json.JSONDecodeError),
        ('{"ids": ["a"]}', KeyError),
        ('{"ids": ["a", "b"], "tokens": [["x"]]}', ValueError),
    ],
    ids=["garbage", "missing-tokens", "length-mismatch"],
)
def test_loading_a_corrupt_file_raises(tmp_path: Path, content: str, error: type) -> None:
    path = tmp_path / "bm25.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(error):
        BM25Index.load(path)


# =====================================================================
# The real mini corpus
# =====================================================================

MINI_CORPUS = Path(__file__).parents[1] / "fixtures" / "mini_corpus"


@pytest.fixture(scope="module")
def corpus() -> tuple[BM25Index, dict[str, Chunk]]:
    chunks = [
        chunk
        for page in list_pages(MINI_CORPUS)
        for chunk in chunk_page(
            clean_mdx(page.read_text(encoding="utf-8")),
            path_to_url(page, MINI_CORPUS),
            page.relative_to(MINI_CORPUS).as_posix(),
        )
    ]
    return BM25Index.build(chunks), {c.id: c for c in chunks}


@pytest.mark.parametrize(
    ("query", "expected_page"),
    [
        ("createPortal", "reference/react-dom/createPortal.md"),
        ("useReducer dispatch", "reference/react/useReducer.md"),
        ("componentDidMount", "reference/react/Component.md"),
        ("useCallback", "reference/react/useCallback.md"),
    ],
)
def test_real_corpus_api_names_find_their_page(
    corpus: tuple[BM25Index, dict[str, Chunk]], query: str, expected_page: str
) -> None:
    index, by_id = corpus
    top_id, _ = index.search(query, k=1)[0]
    assert by_id[top_id].source_path == expected_page
