"""Tests for the abstain gate in the eval harness: choose_threshold and the
abstain metrics that eval/run_eval.py reports.

Retriever and reranker are faked, so everything runs offline in milliseconds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from codesearch.config import Settings
from codesearch.schemas import Chunk, ScoredChunk
from eval.run_eval import ThresholdChoice, choose_threshold, run

# =====================================================================
# choose_threshold
# =====================================================================

IN_SCOPE = [0.80, 0.70, 0.65]
OUT_OF_SCOPE = [0.50, 0.55, 0.60, 0.72]


def test_picks_the_lowest_threshold_that_reaches_the_target() -> None:
    # 3 of 4 out-of-scope scores must be below t: the smallest candidate that does it is 0.65
    choice = choose_threshold(IN_SCOPE, OUT_OF_SCOPE, target_recall=0.75)
    assert isinstance(choice, ThresholdChoice)
    assert choice.threshold == pytest.approx(0.65)
    assert choice.abstain_recall == pytest.approx(0.75)
    assert choice.false_abstain_rate == 0.0  # 0.65 itself answers (score >= threshold)


def test_a_stricter_target_costs_in_scope_answers() -> None:
    # all 4 out-of-scope must abstain, so t must be above 0.72: the next score is 0.80
    choice = choose_threshold(IN_SCOPE, OUT_OF_SCOPE, target_recall=1.0)
    assert choice.threshold == pytest.approx(0.80)
    assert choice.abstain_recall == 1.0
    assert choice.false_abstain_rate == pytest.approx(2 / 3)  # 0.70 and 0.65 now refused


def test_threshold_goes_just_above_the_highest_score_when_needed() -> None:
    choice = choose_threshold([0.4], [0.9], target_recall=1.0)
    assert choice.threshold > 0.9
    assert choice.threshold == pytest.approx(0.9)
    assert choice.abstain_recall == 1.0
    assert choice.false_abstain_rate == 1.0


def test_perfectly_separated_scores_give_no_false_abstains() -> None:
    choice = choose_threshold([0.8, 0.9], [0.3, 0.4], target_recall=0.9)
    assert choice.threshold == pytest.approx(0.8)
    assert (choice.abstain_recall, choice.false_abstain_rate) == (1.0, 0.0)


def test_default_target_is_ninety_percent() -> None:
    oos = [i / 10 for i in range(10)]  # 0.0 .. 0.9
    choice = choose_threshold([1.0], oos)
    assert choice.abstain_recall >= 0.9
    assert choice.threshold == pytest.approx(0.9)  # 9 of 10 below 0.9


def test_works_with_negative_rerank_logits() -> None:
    choice = choose_threshold([3.0, 1.5], [-4.0, -2.0, 0.5], target_recall=1.0)
    assert choice.threshold == pytest.approx(1.5)
    assert choice.false_abstain_rate == 0.0


def test_duplicate_scores_are_handled() -> None:
    choice = choose_threshold([0.7, 0.7], [0.5, 0.5, 0.5], target_recall=1.0)
    assert choice.threshold == pytest.approx(0.7)
    assert choice.false_abstain_rate == 0.0


def test_no_in_scope_scores_means_no_false_abstains() -> None:
    choice = choose_threshold([], [0.2, 0.3], target_recall=1.0)
    assert choice.false_abstain_rate == 0.0
    assert choice.abstain_recall == 1.0


def test_needs_out_of_scope_scores() -> None:
    with pytest.raises(ValueError):
        choose_threshold([0.8], [])


@pytest.mark.parametrize("target", [0.0, -0.1, 1.01])
def test_target_must_be_in_zero_one(target: float) -> None:
    with pytest.raises(ValueError):
        choose_threshold([0.8], [0.3], target_recall=target)


def test_values_are_plain_floats() -> None:
    choice = choose_threshold(IN_SCOPE, OUT_OF_SCOPE)
    assert all(
        type(v) is float
        for v in (choice.threshold, choice.abstain_recall, choice.false_abstain_rate)
    )
    assert math.isfinite(choice.threshold)


# =====================================================================
# run(): the abstain gate on every golden row
# =====================================================================

A = "https://react.dev/reference/react/useState"
B = "https://react.dev/learn/state-a-components-memory"

GOLDEN = [
    {
        "id": "q001",
        "question": "Q1?",
        "expected_urls": [f"{A}#a"],
        "category": "hooks",
        "expected": "answer",
    },
    {
        "id": "q002",
        "question": "Q2?",
        "expected_urls": [f"{B}#b"],
        "category": "state",
        "expected": "answer",
    },
    {
        "id": "oos001",
        "question": "Vue?",
        "expected_urls": [],
        "category": "oos_frontend",
        "expected": "abstain",
    },
    {
        "id": "oos002",
        "question": "SQL?",
        "expected_urls": [],
        "category": "oos_general",
        "expected": "abstain",
    },
]
# best dense score per question: q001 0.9, q002 0.55 (weak), Vue 0.5, SQL 0.7 (looks relevant)
TOP = {"Q1?": 0.9, "Q2?": 0.55, "Vue?": 0.5, "SQL?": 0.7}


def chunk(url_with_anchor: str, dense: float | None = None, **scores: Any) -> ScoredChunk:
    url, _, anchor = url_with_anchor.partition("#")
    c = Chunk(
        id=url_with_anchor,
        url=url,
        anchor=anchor,
        title="t",
        header_path=(),
        text="x",
        source_path="x.md",
        start_line=0,
    )
    return ScoredChunk(chunk=c, dense_score=dense, **scores)


class DenseFake:
    def __init__(self, top: dict[str, float]) -> None:
        self.top = top

    def retrieve(self, query: str) -> list[ScoredChunk]:
        best = self.top[query]
        return [chunk(f"{A}#a", best, dense_rank=1), chunk(f"{B}#x", best - 0.1, dense_rank=2)]


class EmptyFake:
    def retrieve(self, query: str) -> list[ScoredChunk]:
        return []


class Bm25OnlyFake:
    def retrieve(self, query: str) -> list[ScoredChunk]:
        return [chunk(f"{A}#a", None, bm25_rank=1, bm25_score=7.0)]


def run_with(tmp_path: Path, retriever: Any, threshold: float = 0.6, extra: str = "") -> Any:
    golden = tmp_path / "golden.jsonl"
    golden.write_text("".join(json.dumps(r) + "\n" for r in GOLDEN), encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"name: gate\noverrides:\n  retrieval:\n    mode: dense\n"
        f"    abstain_threshold: {threshold}\n{extra}",
        encoding="utf-8",
    )
    settings = Settings(_env_file=None).with_overrides(  # type: ignore[call-arg]
        index={"index_dir": tmp_path / "no-index", "embedding_model": "hash"}
    )
    return run(
        cfg,
        golden_path=golden,
        settings=settings,
        retriever=retriever,
        results_dir=tmp_path / "results",
        log_mlflow=False,
    )


def test_every_row_gets_a_top_score_and_a_decision(tmp_path: Path) -> None:
    result = run_with(tmp_path, DenseFake(TOP))
    answer = {r["id"]: r for r in result.rows}
    oos = {r["id"]: r for r in result.abstain_rows}
    assert set(answer) == {"q001", "q002"}
    assert set(oos) == {"oos001", "oos002"}
    assert answer["q001"]["top_score"] == pytest.approx(0.9)
    assert answer["q001"]["abstained"] is False
    assert answer["q002"]["abstained"] is True  # 0.55 < 0.6
    assert oos["oos001"]["abstained"] is True
    assert oos["oos002"]["abstained"] is False  # 0.7 slips through
    assert oos["oos002"]["score_kind"] == "dense"


def test_abstain_metrics(tmp_path: Path) -> None:
    m = run_with(tmp_path, DenseFake(TOP)).metrics
    # abstained on: q002 (wrong), oos001 (right). should abstain: oos001, oos002
    assert m["abstain_precision"] == pytest.approx(0.5)
    assert m["abstain_recall"] == pytest.approx(0.5)
    assert m["false_abstain_rate"] == pytest.approx(0.5)  # 1 of 2 in-scope refused


def test_retrieval_metrics_ignore_the_abstain_decision(tmp_path: Path) -> None:
    """q002 is refused, but its retrieval is still scored: the gate is measured separately."""
    result = run_with(tmp_path, DenseFake(TOP))
    assert result.rows[1]["scores"]["page_recall_at_5"] == pytest.approx(1.0)


def test_threshold_comes_from_settings(tmp_path: Path) -> None:
    low = run_with(tmp_path, DenseFake(TOP), threshold=0.0).metrics
    assert low["false_abstain_rate"] == 0.0
    assert low["abstain_recall"] == 0.0
    high = run_with(tmp_path, DenseFake(TOP), threshold=0.95).metrics
    assert high["abstain_recall"] == 1.0
    assert high["false_abstain_rate"] == 1.0


def test_abstain_params_and_suggested_threshold_are_recorded(tmp_path: Path) -> None:
    result = run_with(tmp_path, DenseFake(TOP))
    assert result.params["abstain_threshold"] == pytest.approx(0.6)
    assert result.params["n_out_of_scope"] == 2
    # choose_threshold over the in-scope [0.9, 0.55] and out-of-scope [0.5, 0.7] top scores
    expected = choose_threshold([0.9, 0.55], [0.5, 0.7], target_recall=0.9)
    assert result.suggested_threshold == expected


def test_results_file_holds_the_abstain_rows(tmp_path: Path) -> None:
    result = run_with(tmp_path, DenseFake(TOP))
    data = json.loads(result.results_path.read_text(encoding="utf-8"))
    assert {r["id"] for r in data["abstain_rows"]} == {"oos001", "oos002"}
    assert data["suggested_threshold"]["threshold"] == pytest.approx(
        result.suggested_threshold.threshold
    )


def test_nothing_retrieved_abstains_with_no_score(tmp_path: Path) -> None:
    result = run_with(tmp_path, EmptyFake())
    assert all(r["abstained"] is True and r["top_score"] is None for r in result.abstain_rows)
    assert result.metrics["abstain_recall"] == 1.0
    assert result.suggested_threshold is None  # no scores to tune on


def test_bm25_only_results_skip_the_gate(tmp_path: Path) -> None:
    """No rerank/dense score: the gate can't judge, so no decision and no abstain metrics."""
    result = run_with(tmp_path, Bm25OnlyFake())
    assert all(r["abstained"] is None for r in result.rows + result.abstain_rows)
    assert "abstain_precision" not in result.metrics
    assert result.suggested_threshold is None


def test_rerank_scores_are_used_for_the_gate_after_reranking(tmp_path: Path) -> None:
    class Rerank:
        def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
            logit = {"Q1?": 3.0, "Q2?": 1.0, "Vue?": -5.0, "SQL?": -1.0}[query]
            return [c.model_copy(update={"rerank_score": logit}) for c in chunks][:top_n]

    golden = tmp_path / "golden.jsonl"
    golden.write_text("".join(json.dumps(r) + "\n" for r in GOLDEN), encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "name: rr\noverrides:\n  retrieval:\n    mode: dense\n    reranker_model: m\n"
        "    rerank_top_n: 5\n    abstain_threshold: 0.0\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=None).with_overrides(  # type: ignore[call-arg]
        index={"index_dir": tmp_path / "no-index", "embedding_model": "hash"}
    )
    result = run(
        cfg,
        golden_path=golden,
        settings=settings,
        retriever=DenseFake(TOP),
        reranker=Rerank(),
        results_dir=tmp_path / "r",
        log_mlflow=False,
    )
    assert all(r["score_kind"] == "rerank" for r in result.rows + result.abstain_rows)
    assert result.metrics["abstain_recall"] == 1.0
    assert result.metrics["false_abstain_rate"] == 0.0
