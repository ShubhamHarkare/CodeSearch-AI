"""Tests for eval/fit_calibrator.py: fit the confidence calibrator on answer-eval results.

uv run python -m eval.answer_eval eval/configs/dense.yaml      # all 79 questions first
uv run python -m eval.fit_calibrator results/answers_dense.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codesearch.explain.calibration import Calibrator
from eval.fit_calibrator import dataset, is_good, run


def record(
    good: bool | None = True,
    *,
    expected: str = "answer",
    abstained: bool = False,
    features: list[float] | None = None,
) -> dict[str, Any]:
    """good=True: cites a right page and is well grounded; False: neither."""
    return {
        "expected": expected,
        "abstained": abstained,
        "cited_relevant": bool(good),
        "groundedness": 0.9 if good else 0.1,
        "features": features
        if features is not None
        else ([0.9, 0.85, 0.9] if good else [0.5, 0.4, 0.1]),
    }


# =====================================================================
# is_good: the label the calibrator learns to predict
# =====================================================================


def test_a_good_answer_cites_a_right_page_and_is_mostly_grounded() -> None:
    assert is_good({"cited_relevant": True, "groundedness": 0.5})
    assert is_good({"cited_relevant": True, "groundedness": 1.0})


@pytest.mark.parametrize(
    "rec",
    [
        {"cited_relevant": False, "groundedness": 1.0},  # wrong page
        {"cited_relevant": None, "groundedness": 1.0},  # cited nothing
        {"cited_relevant": True, "groundedness": 0.49},  # mostly unsupported
        {"cited_relevant": True, "groundedness": None},  # never graded
    ],
)
def test_anything_else_is_not_good(rec: dict[str, Any]) -> None:
    assert not is_good(rec)


# =====================================================================
# dataset: which records train the calibrator
# =====================================================================


def test_dataset_uses_answered_in_scope_questions_with_features() -> None:
    records = [
        record(True),
        record(False),
        record(True, expected="abstain", abstained=True),  # out of scope: no confidence shown
        record(True, abstained=True),  # refused: no answer to be confident about
        {**record(True), "features": None},  # no trace: no features
    ]
    X, y = dataset(records)
    assert X == [[0.9, 0.85, 0.9], [0.5, 0.4, 0.1]]
    assert y == [1, 0]


# =====================================================================
# run: fit, check, save
# =====================================================================


def write_answers(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps({"config": "dense", "metrics": {}, "rows": records}))
    return path


def test_run_fits_saves_and_reports(tmp_path: Path) -> None:
    records = [record(True)] * 6 + [record(False)] * 6
    answers = write_answers(tmp_path / "answers_dense.json", records)
    out = tmp_path / "models" / "calibrator.json"

    report = run(answers, out, n_bins=5)

    calibrator = Calibrator.load(out)
    assert calibrator.confidence([0.9, 0.85, 0.9]) > calibrator.confidence([0.5, 0.4, 0.1])
    assert report["n"] == 12
    assert report["positive_rate"] == 0.5
    assert 0.0 <= report["ece"] <= 1.0
    assert sum(b["count"] for b in report["bins"]) == 12
    assert set(report["bins"][0]) == {"lower", "upper", "count", "confidence", "accuracy"}


def test_run_reports_out_of_fold_calibration(tmp_path: Path) -> None:
    """The ECE is measured on predictions for questions the model wasn't fitted on,
    so it isn't flattered by checking the model on its own training data."""
    records = [record(True)] * 6 + [record(False)] * 6
    report = run(
        write_answers(tmp_path / "a.json", records), tmp_path / "c.json", n_bins=5, folds=3
    )
    assert report["folds"] == 3


def test_run_refuses_a_set_with_only_one_kind_of_answer(tmp_path: Path) -> None:
    answers = write_answers(tmp_path / "a.json", [record(True)] * 5)
    with pytest.raises(ValueError):
        run(answers, tmp_path / "c.json")
