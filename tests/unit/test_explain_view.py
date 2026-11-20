"""Tests for codesearch/ui/explain_view.py: turns a trace (the JSON from
GET /v1/explain/{trace_id}) into what the "Why this answer?" panel shows.

Pure functions, no Streamlit, so the panel's logic is tested without a browser.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from codesearch.ui.explain_view import (
    attribution_rows,
    confidence_badge,
    format_matched,
    format_ms,
    grounding_note,
    message_from_answer,
    sentence_rows,
    source_rows,
    sources_line,
    summary_lines,
    timing_caption,
)


def candidate(chunk_id: str, title: str = "", **scores: float | None) -> dict[str, Any]:
    return {
        "chunk": {
            "id": chunk_id,
            "title": title or chunk_id,
            "url": f"https://react.dev/{chunk_id}",
        },
        "dense_score": scores.get("dense_score"),
        "bm25_score": scores.get("bm25_score"),
        "rrf_score": scores.get("rrf_score"),
        "rerank_score": scores.get("rerank_score"),
    }


def trace(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "trace_id": "t-1",
        "question": "How do I update state?",
        "standalone_question": "How do I update state?",
        "candidates": [],
        "used_chunk_ids": [],
        "matched_terms": {},
        "sentences": [],
        "abstain_reason": None,
        "cache_tier": None,
        "timings": {},
        "versions": {},
    }
    return {**base, **overrides}


# =====================================================================
# format_ms / format_matched
# =====================================================================


@pytest.mark.parametrize(
    ("ms", "text"),
    [(0.4, "0 ms"), (78.2, "78 ms"), (999.4, "999 ms"), (1000.0, "1.0 s"), (8425.5, "8.4 s")],
)
def test_format_ms(ms: float, text: str) -> None:
    assert format_ms(ms) == text


def test_format_matched_sorts_by_weight_and_rounds() -> None:
    assert format_matched({"update": 1.43, "state": 3.07}) == "state 3.1, update 1.4"


def test_format_matched_empty() -> None:
    assert format_matched({}) == ""


# =====================================================================
# source_rows: the score table
# =====================================================================


def test_used_sources_come_first_in_prompt_order_then_the_rest() -> None:
    t = trace(
        candidates=[candidate("a"), candidate("b"), candidate("c"), candidate("d")],
        used_chunk_ids=["c", "a"],
    )
    rows = source_rows(t)
    assert [r["title"] for r in rows] == ["c", "a", "b", "d"]
    assert [r["used"] for r in rows] == [True, True, False, False]


def test_used_sources_are_numbered_like_the_citations() -> None:
    """[1] in the answer is the first chunk sent to the LLM, so number used rows 1, 2, ..."""
    t = trace(
        candidates=[candidate("a"), candidate("b"), candidate("c")], used_chunk_ids=["b", "a"]
    )
    assert [r["n"] for r in source_rows(t)] == [1, 2, None]


def test_rows_carry_scores_rounded_and_none_stays_none() -> None:
    t = trace(
        candidates=[
            candidate("a", "useState", dense_score=0.81234, bm25_score=None, rerank_score=4.56789)
        ],
        used_chunk_ids=["a"],
    )
    row = source_rows(t)[0]
    assert row["title"] == "useState"
    assert row["url"] == "https://react.dev/a"
    assert row["dense"] == 0.812
    assert row["bm25"] is None
    assert row["rerank"] == 4.568
    assert row["rrf"] is None


def test_rows_show_matched_terms() -> None:
    t = trace(
        candidates=[candidate("a"), candidate("b")],
        used_chunk_ids=["a"],
        matched_terms={"a": {"state": 3.07, "update": 1.43}},
    )
    rows = source_rows(t)
    assert rows[0]["matched"] == "state 3.1, update 1.4"
    assert rows[1]["matched"] == ""  # no entry for unused chunks


def test_a_used_id_missing_from_candidates_still_gets_a_row() -> None:
    """An exact cache hit has used ids but no candidates (retrieval never ran)."""
    rows = source_rows(trace(used_chunk_ids=["x"]))
    assert len(rows) == 1
    assert rows[0]["used"] is True and rows[0]["n"] == 1
    assert rows[0]["title"] == "x"  # all we know is the id
    assert rows[0]["dense"] is None


def test_no_candidates_no_rows() -> None:
    assert source_rows(trace()) == []


# =====================================================================
# summary_lines: the short text above the table
# =====================================================================


def test_summary_for_a_plain_answer_is_just_timings() -> None:
    t = trace(timings={"retrieval_ms": 78.2, "generation_ms": 8425.5, "total_ms": 8516.3})
    assert summary_lines(t) == ["retrieval 78 ms · generation 8.4 s · total 8.5 s"]


def test_timings_follow_the_pipeline_order() -> None:
    t = trace(
        timings={
            "total_ms": 9000.0,
            "generation_ms": 8000.0,
            "condense_ms": 500.0,
            "retrieval_ms": 80.0,
        }
    )
    assert (
        summary_lines(t)[-1] == "condense 500 ms · retrieval 80 ms · generation 8.0 s · total 9.0 s"
    )


def test_summary_shows_the_rewritten_question_for_follow_ups() -> None:
    t = trace(
        question="what about objects?", standalone_question="How do I update an object in state?"
    )
    assert "Searched as: How do I update an object in state?" in summary_lines(t)


def test_summary_explains_an_abstain() -> None:
    t = trace(abstain_reason="below_threshold (dense 0.512 < 0.755)")
    assert "Abstained: below_threshold (dense 0.512 < 0.755)" in summary_lines(t)


def test_summary_mentions_a_cache_hit() -> None:
    assert "Served from the exact cache." in summary_lines(trace(cache_tier="exact"))
    assert "Served from the semantic cache." in summary_lines(trace(cache_tier="semantic"))
    shared = summary_lines(trace(cache_tier="shared"))
    assert any("identical question" in line for line in shared)
    assert not any("shared cache" in line for line in shared)


def test_an_empty_trace_has_no_summary() -> None:
    assert summary_lines(trace()) == []


def test_explain_view_does_not_import_streamlit_or_the_pipeline() -> None:
    import codesearch.ui.explain_view as module

    tree = ast.parse(Path(module.__file__).read_text())
    names = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not any(name.split(".")[0] in {"streamlit", "codesearch"} for name in names)


# =====================================================================
# sentence_rows / grounding_note: groundedness, made visible
# =====================================================================


def sentence(text: str, label: str, score: float, *ids: str) -> dict[str, Any]:
    return {"sentence": text, "cited_chunk_ids": list(ids), "label": label, "score": score}


def test_sentence_rows_label_each_sentence_and_name_its_sources() -> None:
    t = trace(
        used_chunk_ids=["c", "a"],
        sentences=[
            sentence("Use an updater [1].", "supported", 0.912, "c"),
            sentence("It runs later [2].", "partial", 0.55, "a"),
            sentence("React is the best.", "unsupported", 0.0),
        ],
    )
    assert sentence_rows(t) == [
        {"label": "supported", "score": 0.91, "sentence": "Use an updater [1].", "sources": "[1]"},
        {"label": "partial", "score": 0.55, "sentence": "It runs later [2].", "sources": "[2]"},
        {"label": "unsupported", "score": 0.0, "sentence": "React is the best.", "sources": ""},
    ]


def test_sentence_rows_number_sources_like_the_citations() -> None:
    """A sentence citing two chunks shows both citation numbers, in prompt order."""
    t = trace(
        used_chunk_ids=["x", "y", "z"],
        sentences=[sentence("Both [1, 3].", "supported", 0.8, "z", "x")],
    )
    assert sentence_rows(t)[0]["sources"] == "[1, 3]"


def test_sentence_rows_without_sentences() -> None:
    assert sentence_rows(trace()) == []


@pytest.mark.parametrize(
    ("labels", "note"),
    [
        (["supported", "supported"], ""),
        (["supported", "unsupported"], "1 of 2 sentences isn't supported by its source"),
        (
            ["unsupported", "partial", "unsupported"],
            "2 of 3 sentences aren't supported by their sources",
        ),
        (["supported", "partial"], "1 of 2 sentences is only partly supported"),
        ([], ""),
    ],
)
def test_grounding_note_flags_weak_sentences(labels: list[str], note: str) -> None:
    """Shown under the answer: unsupported sentences are the warning that matters most."""
    t = trace(sentences=[sentence(f"S{i}.", label, 0.5) for i, label in enumerate(labels)])
    assert grounding_note(t) == note


# =====================================================================
# confidence_badge: the calibrated confidence, as a high/medium/low badge
# =====================================================================


@pytest.mark.parametrize(
    ("confidence", "badge"),
    [
        (0.91, ("high", "green", "high confidence 91%")),
        (0.75, ("high", "green", "high confidence 75%")),
        (0.74, ("medium", "orange", "medium confidence 74%")),
        (0.40, ("medium", "orange", "medium confidence 40%")),
        (0.12, ("low", "red", "low confidence 12%")),
    ],
)
def test_confidence_badge_levels(confidence: float, badge: tuple[str, str, str]) -> None:
    """Same thresholds as Calibrator.level (0.75 / 0.4), so UI and API agree."""
    assert confidence_badge(confidence) == badge


def test_no_confidence_no_badge() -> None:
    """No calibrator fitted yet, or an abstained answer."""
    assert confidence_badge(None) is None


def test_badge_thresholds_match_the_calibrator_defaults() -> None:
    from codesearch.explain.calibration import Calibrator

    cal = Calibrator(coef=[0.0, 0.0, 0.0], intercept=0.0)
    for c in (0.0, 0.39, 0.4, 0.6, 0.749, 0.75, 1.0):
        badge = confidence_badge(c)
        assert badge is not None and badge[0] == cal.level(c)


# =====================================================================
# attribution_rows: the Deep explain result, one row per source
# =====================================================================


def test_attribution_rows_follow_the_citation_numbers() -> None:
    t = trace(
        candidates=[
            candidate("a", "useState"),
            candidate("b", "Queueing"),
            candidate("c", "Effects"),
        ],
        used_chunk_ids=["b", "a", "c"],
    )
    rows = attribution_rows(t, {"a": 0.25, "b": 0.75, "c": 0.0})
    assert [(r["n"], r["title"], r["share"]) for r in rows] == [
        (1, "Queueing", 0.75),
        (2, "useState", 0.25),
        (3, "Effects", 0.0),
    ]
    assert rows[0]["url"] == "https://react.dev/b"


def test_attribution_rows_missing_score_counts_as_zero() -> None:
    t = trace(candidates=[candidate("a"), candidate("b")], used_chunk_ids=["a", "b"])
    assert [r["share"] for r in attribution_rows(t, {"a": 1.0})] == [1.0, 0.0]


def test_attribution_rows_only_cover_sources_sent_to_the_model() -> None:
    t = trace(candidates=[candidate("a"), candidate("b")], used_chunk_ids=["a"])
    assert [r["title"] for r in attribution_rows(t, {"a": 1.0, "b": 0.0})] == ["a"]


# =====================================================================
# chat message helpers (streaming UI)
# =====================================================================

ASK_RESPONSE = {
    "answer": "Use an updater [1].",
    "citations": [{"number": 1, "chunk_id": "c", "url": "https://react.dev/a", "title": "A"}],
    "abstained": False,
    "cache_tier": None,
    "timings": {"total_ms": 6900.0},
    "trace_id": "t",
    "groundedness": 0.8,
    "confidence": 0.7,
}


def test_message_from_ask_and_from_stream_done_are_the_same() -> None:
    done = {k: v for k, v in ASK_RESPONSE.items() if k != "answer"} | {
        "text": "Use an updater [1]."
    }
    assert message_from_answer(done) == message_from_answer(ASK_RESPONSE)
    msg = message_from_answer(ASK_RESPONSE)
    assert msg["role"] == "assistant"
    assert msg["content"] == "Use an updater [1]."
    assert (msg["total_ms"], msg["ttft_ms"], msg["trace_id"]) == (6900.0, None, "t")


def test_message_from_answer_keeps_ttft_when_streamed() -> None:
    done = {"text": "x", "timings": {"total_ms": 6900.0, "ttft_ms": 850.0}, "trace_id": "t"}
    assert message_from_answer(done)["ttft_ms"] == 850.0


def test_sources_line_numbers_and_links_each_source() -> None:
    sources = [
        {"number": 1, "url": "https://react.dev/a", "title": "A"},
        {"number": 2, "url": "https://react.dev/b", "title": "B"},
    ]
    assert sources_line(sources) == "[1] [A](https://react.dev/a) · [2] [B](https://react.dev/b)"
    assert sources_line([]) == ""


@pytest.mark.parametrize(
    ("total", "ttft", "text"),
    [(6900.0, 850.0, "first words 0.8s · 6.9s"), (6900.0, None, "6.9s"), (None, None, "")],
)
def test_timing_caption(total: float | None, ttft: float | None, text: str) -> None:
    assert timing_caption(total, ttft) == text
