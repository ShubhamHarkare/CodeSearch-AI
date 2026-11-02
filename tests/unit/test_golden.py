"""Tests for eval/golden.py: the golden-set row model, loader and URL check.

Three layers:
  1. GoldenRow rules (pure, hand-made rows)
  2. load_golden / missing_urls on small temp files
  3. the real eval/golden.jsonl: schema in CI, URLs against the local index (skipped in CI)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from eval.golden import GoldenRow, load_golden, missing_urls

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "eval" / "golden.jsonl"
INDEX_DIR = REPO_ROOT / "index"

URL = "https://react.dev/reference/react/useState#usestate"
URL2 = "https://react.dev/learn/state-a-components-memory"


def answer_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "q001",
        "question": "How do I use the useState hook?",
        "expected_urls": [URL],
        "category": "hooks",
        "expected": "answer",
    }
    row.update(overrides)
    return row


def abstain_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "oos001",
        "question": "How do I write a computed property in Vue?",
        "expected_urls": [],
        "category": "oos_frontend",
        "expected": "abstain",
    }
    row.update(overrides)
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# =====================================================================
# GoldenRow
# =====================================================================


def test_valid_answer_row() -> None:
    row = GoldenRow.model_validate(answer_row())
    assert row.id == "q001"
    assert row.expected == "answer"
    assert list(row.expected_urls) == [URL]
    assert row.notes is None


def test_valid_abstain_row() -> None:
    row = GoldenRow.model_validate(abstain_row())
    assert row.expected == "abstain"
    assert len(row.expected_urls) == 0


def test_notes_are_optional_and_kept() -> None:
    row = GoldenRow.model_validate(answer_row(notes="answer is in a deep dive"))
    assert row.notes == "answer is in a deep dive"


def test_page_level_url_without_anchor_is_allowed() -> None:
    assert GoldenRow.model_validate(answer_row(expected_urls=[URL2]))


def test_row_is_frozen() -> None:
    row = GoldenRow.model_validate(answer_row())
    with pytest.raises(ValidationError):
        row.question = "changed"


@pytest.mark.parametrize("field", ["id", "question", "expected_urls", "category", "expected"])
def test_required_fields(field: str) -> None:
    data = answer_row()
    del data[field]
    with pytest.raises(ValidationError):
        GoldenRow.model_validate(data)


def test_unknown_fields_are_rejected() -> None:
    """A typo like 'expected_url' must fail loudly, not be silently dropped."""
    with pytest.raises(ValidationError):
        GoldenRow.model_validate(answer_row(expected_url=[URL]))


@pytest.mark.parametrize("value", ["maybe", "Answer", "", None])
def test_expected_must_be_answer_or_abstain(value: Any) -> None:
    with pytest.raises(ValidationError):
        GoldenRow.model_validate(answer_row(expected=value))


@pytest.mark.parametrize("field", ["id", "question", "category"])
@pytest.mark.parametrize("value", ["", "   "])
def test_text_fields_must_not_be_blank(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        GoldenRow.model_validate(answer_row(**{field: value}))


def test_answer_row_needs_at_least_one_url() -> None:
    with pytest.raises(ValidationError):
        GoldenRow.model_validate(answer_row(expected_urls=[]))


def test_abstain_row_must_have_no_urls() -> None:
    with pytest.raises(ValidationError):
        GoldenRow.model_validate(abstain_row(expected_urls=[URL]))


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://react.dev/learn",  # not https
        "https://legacy.reactjs.org/docs/hooks-state.html",  # old docs site
        "https://react.dev.evil.com/learn",  # prefix trick
        "react.dev/learn",  # no scheme
        "/learn/state-a-components-memory",  # relative
        "",
    ],
)
def test_urls_must_be_react_dev(bad_url: str) -> None:
    with pytest.raises(ValidationError):
        GoldenRow.model_validate(answer_row(expected_urls=[bad_url]))


def test_duplicate_urls_in_one_row_are_rejected() -> None:
    with pytest.raises(ValidationError):
        GoldenRow.model_validate(answer_row(expected_urls=[URL, URL]))


# =====================================================================
# load_golden
# =====================================================================


def test_load_golden_reads_rows_in_file_order(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "g.jsonl", [answer_row(), abstain_row()])
    rows = load_golden(path)
    assert [r.id for r in rows] == ["q001", "oos001"]
    assert all(isinstance(r, GoldenRow) for r in rows)


def test_load_golden_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text(
        "\n" + json.dumps(answer_row()) + "\n   \n\n" + json.dumps(abstain_row()) + "\n\n",
        encoding="utf-8",
    )
    assert [r.id for r in load_golden(path)] == ["q001", "oos001"]


def test_load_golden_without_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(answer_row()), encoding="utf-8")
    assert len(load_golden(path)) == 1


def test_load_golden_empty_file_gives_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text("", encoding="utf-8")
    assert load_golden(path) == []


def test_load_golden_handles_non_ascii(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "g.jsonl", [answer_row(question="Wie benutze ich useState? ✓")])
    assert load_golden(path)[0].question.endswith("✓")


def test_load_golden_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_golden(tmp_path / "nope.jsonl")


def test_bad_json_reports_the_line_number(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(answer_row()) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2"):
        load_golden(path)


def test_invalid_row_reports_the_line_number(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "g.jsonl",
        [answer_row(), abstain_row(), answer_row(id="q002", expected_urls=[])],
    )
    with pytest.raises(ValueError, match=r"line 3"):
        load_golden(path)


def test_line_numbers_count_blank_lines(tmp_path: Path) -> None:
    """Line numbers must match what an editor shows, so blank lines still count."""
    path = tmp_path / "g.jsonl"
    path.write_text("\n\n" + json.dumps(answer_row(expected="nope")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 3"):
        load_golden(path)


def test_a_json_value_that_is_not_an_object_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text('["q001", "How?"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 1"):
        load_golden(path)


def test_duplicate_ids_are_rejected_and_named(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "g.jsonl",
        [answer_row(), abstain_row(), answer_row(question="Something else?")],
    )
    with pytest.raises(ValueError, match=r"q001"):
        load_golden(path)


# =====================================================================
# missing_urls
# =====================================================================


def test_missing_urls_lists_only_unknown_urls_per_row() -> None:
    rows = [
        GoldenRow.model_validate(answer_row(id="q001", expected_urls=[URL, URL2])),
        GoldenRow.model_validate(answer_row(id="q002", expected_urls=[URL])),
    ]
    assert missing_urls(rows, {URL}) == {"q001": [URL2]}


def test_missing_urls_is_empty_when_everything_is_known() -> None:
    rows = [GoldenRow.model_validate(answer_row(expected_urls=[URL, URL2]))]
    assert missing_urls(rows, [URL, URL2]) == {}


def test_missing_urls_ignores_abstain_rows() -> None:
    rows = [GoldenRow.model_validate(abstain_row())]
    assert missing_urls(rows, set()) == {}


def test_missing_urls_keeps_url_order() -> None:
    other = "https://react.dev/learn#a"
    rows = [GoldenRow.model_validate(answer_row(expected_urls=[URL2, other, URL]))]
    assert missing_urls(rows, set()) == {"q001": [URL2, other, URL]}


def test_missing_urls_requires_the_exact_anchor() -> None:
    """Right page, wrong anchor is still missing: section-level ground truth must be exact."""
    rows = [GoldenRow.model_validate(answer_row(expected_urls=[URL]))]
    assert missing_urls(rows, {"https://react.dev/reference/react/useState"}) == {"q001": [URL]}


def test_missing_urls_with_no_rows() -> None:
    assert missing_urls([], {URL}) == {}


# =====================================================================
# The real golden set (data checks)
# =====================================================================


@pytest.fixture(scope="module")
def golden() -> list[GoldenRow]:
    return load_golden(GOLDEN_PATH)


def test_real_golden_set_loads(golden: list[GoldenRow]) -> None:
    """Runs in CI: every row parses and passes the GoldenRow rules, ids are unique."""
    assert len(golden) > 0


def test_real_golden_set_has_both_kinds(golden: list[GoldenRow]) -> None:
    kinds = Counter(r.expected for r in golden)
    assert kinds["answer"] >= 20
    assert kinds["abstain"] >= 10


def test_real_golden_ids_match_their_kind(golden: list[GoldenRow]) -> None:
    for r in golden:
        if r.expected == "answer":
            assert r.id.startswith("q"), r.id
        else:
            assert r.id.startswith("oos"), r.id
            assert r.category.startswith("oos_"), r.id


def test_real_golden_questions_are_unique(golden: list[GoldenRow]) -> None:
    counts = Counter(r.question.strip().lower() for r in golden)
    assert [q for q, n in counts.items() if n > 1] == []


def test_real_golden_rows_have_at_most_three_urls(golden: list[GoldenRow]) -> None:
    assert [r.id for r in golden if len(r.expected_urls) > 3] == []


def test_real_golden_urls_exist_in_the_local_index(golden: list[GoldenRow]) -> None:
    """Every expected URL must be a real chunk url#anchor. Needs a built index, so CI skips it."""
    if not (INDEX_DIR / "manifest.json").exists():
        pytest.skip("no local index; run `codesearch ingest` first")
    from codesearch.index.handle import CHROMA_DIR
    from codesearch.index.vector_store import VectorStore
    from codesearch.ingest.urls import url_with_anchor

    store = VectorStore(INDEX_DIR / CHROMA_DIR)
    known = {url_with_anchor(c.url, c.anchor) for c in store.get(sorted(store.all_ids()))}
    assert missing_urls(golden, known) == {}
