"""Tests for codesearch/explain/calibration.py: a confidence score you can trust.

features() turns an answer into a few numbers; Calibrator maps them to a probability
that the answer is good (logistic regression, saved as plain JSON); reliability_bins()
and expected_calibration_error() check that "80% confident" means right ~80% of the time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codesearch.explain.calibration import (
    FEATURE_NAMES,
    Calibrator,
    expected_calibration_error,
    features,
    reliability_bins,
)
from codesearch.schemas import Chunk, ScoredChunk


def scored(dense: float | None = None, rerank: float | None = None) -> ScoredChunk:
    chunk = Chunk(
        id=f"c{dense}{rerank}",
        url="https://react.dev/x",
        anchor="",
        title="x",
        header_path=("x",),
        text="t",
        source_path="x.md",
        start_line=1,
    )
    return ScoredChunk(chunk=chunk, dense_score=dense, rerank_score=rerank)


# =====================================================================
# features
# =====================================================================


def test_feature_names_match_the_feature_vector() -> None:
    assert FEATURE_NAMES == ("top_score", "mean_top3", "groundedness")
    assert len(features([scored(0.8)], 0.5)) == len(FEATURE_NAMES)


def test_features_use_dense_scores_when_nothing_was_reranked() -> None:
    chunks = [scored(0.7), scored(0.9), scored(0.8), scored(0.1)]
    top, mean3, grounded = features(chunks, 0.75)
    assert top == pytest.approx(0.9)
    assert mean3 == pytest.approx((0.9 + 0.8 + 0.7) / 3)  # the best three, any order
    assert grounded == 0.75


def test_features_prefer_rerank_scores_like_the_abstain_gate() -> None:
    chunks = [scored(0.9, rerank=2.0), scored(0.5, rerank=6.0)]
    top, mean3, _ = features(chunks, 1.0)
    assert top == 6.0
    assert mean3 == pytest.approx(4.0)  # fewer than three: mean of what there is


def test_missing_groundedness_counts_as_zero() -> None:
    assert features([scored(0.8)], None)[2] == 0.0


def test_no_chunks_gives_zero_scores() -> None:
    assert features([], 0.5) == [0.0, 0.0, 0.5]


# =====================================================================
# Calibrator
# =====================================================================


def separable() -> tuple[list[list[float]], list[int]]:
    """Good answers have high scores, bad ones low: easy to separate."""
    good = [[0.9, 0.85, 1.0], [0.88, 0.8, 0.9], [0.95, 0.9, 1.0], [0.85, 0.8, 0.8]]
    bad = [[0.5, 0.4, 0.0], [0.55, 0.45, 0.2], [0.45, 0.4, 0.1], [0.6, 0.5, 0.3]]
    return good + bad, [1] * len(good) + [0] * len(bad)


def test_fit_separates_good_from_bad() -> None:
    X, y = separable()
    cal = Calibrator.fit(X, y)
    assert cal.confidence([0.92, 0.88, 1.0]) > 0.7
    assert cal.confidence([0.5, 0.4, 0.0]) < 0.3


def test_confidence_is_a_probability() -> None:
    cal = Calibrator.fit(*separable())
    for x in ([100.0, 100.0, 100.0], [-100.0, -100.0, -100.0], [0.0, 0.0, 0.0]):
        assert 0.0 <= cal.confidence(x) <= 1.0
        assert isinstance(cal.confidence(x), float)


def test_confidence_is_the_logistic_of_the_linear_score() -> None:
    cal = Calibrator(coef=[1.0, 0.0, 0.0], intercept=0.0)
    assert cal.confidence([0.0, 5.0, 5.0]) == pytest.approx(0.5)
    assert cal.confidence([2.0, 0.0, 0.0]) == pytest.approx(1 / (1 + 2.718281828**-2))


def test_confidence_needs_one_value_per_feature() -> None:
    cal = Calibrator(coef=[1.0, 0.0, 0.0], intercept=0.0)
    with pytest.raises(ValueError):
        cal.confidence([1.0, 2.0])


def test_fit_needs_both_good_and_bad_examples() -> None:
    with pytest.raises(ValueError):
        Calibrator.fit([[0.9, 0.9, 1.0], [0.8, 0.8, 1.0]], [1, 1])


def test_fit_needs_matching_lengths() -> None:
    with pytest.raises(ValueError):
        Calibrator.fit([[0.9, 0.9, 1.0]], [1, 0])


@pytest.mark.parametrize(
    ("confidence", "level"),
    [
        (0.95, "high"),
        (0.75, "high"),
        (0.74, "medium"),
        (0.4, "medium"),
        (0.39, "low"),
        (0.0, "low"),
    ],
)
def test_level_buckets(confidence: float, level: str) -> None:
    assert Calibrator(coef=[0.0, 0.0, 0.0], intercept=0.0).level(confidence) == level


def test_level_thresholds_are_configurable() -> None:
    cal = Calibrator(coef=[0.0, 0.0, 0.0], intercept=0.0, high_at=0.9, low_at=0.2)
    assert cal.level(0.8) == "medium"
    assert cal.level(0.25) == "medium"


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    cal = Calibrator.fit(*separable())
    path = tmp_path / "models" / "calibrator.json"  # parent folder is created
    cal.save(path)
    loaded = Calibrator.load(path)
    assert loaded == cal
    assert loaded.confidence([0.9, 0.85, 1.0]) == cal.confidence([0.9, 0.85, 1.0])


def test_the_saved_file_is_plain_json_not_pickle(tmp_path: Path) -> None:
    path = tmp_path / "calibrator.json"
    Calibrator(coef=[1.0, 2.0, 3.0], intercept=-1.5).save(path)
    data = json.loads(path.read_text())
    assert data["coef"] == [1.0, 2.0, 3.0]
    assert data["intercept"] == -1.5
    assert data["feature_names"] == list(FEATURE_NAMES)


def test_loading_a_calibrator_for_different_features_fails(tmp_path: Path) -> None:
    """A file fitted on old features must not silently score new ones."""
    path = tmp_path / "calibrator.json"
    path.write_text(json.dumps({"coef": [1.0], "intercept": 0.0, "feature_names": ["top_score"]}))
    with pytest.raises(ValueError):
        Calibrator.load(path)


# =====================================================================
# reliability_bins / expected_calibration_error
# =====================================================================


def test_reliability_bins_group_by_confidence() -> None:
    probs = [0.1, 0.15, 0.85, 0.9, 0.95]
    labels = [0, 0, 1, 1, 0]
    bins = reliability_bins(probs, labels, n_bins=2)  # [0, 0.5) and [0.5, 1]
    assert [(b.lower, b.upper, b.count) for b in bins] == [(0.0, 0.5, 2), (0.5, 1.0, 3)]
    assert bins[0].confidence == pytest.approx(0.125)
    assert bins[0].accuracy == 0.0
    assert bins[1].confidence == pytest.approx(0.9)
    assert bins[1].accuracy == pytest.approx(2 / 3)


def test_reliability_bins_skip_empty_bins_and_include_one_point_zero() -> None:
    bins = reliability_bins([0.05, 1.0], [0, 1], n_bins=10)
    assert [b.count for b in bins] == [1, 1]
    assert bins[-1].upper == 1.0  # a confidence of exactly 1.0 lands in the last bin


def test_reliability_bins_need_matching_lengths() -> None:
    with pytest.raises(ValueError):
        reliability_bins([0.5], [1, 0])


def test_a_perfectly_calibrated_model_has_zero_ece() -> None:
    # every bin's mean confidence equals its accuracy
    probs = [0.25] * 4 + [0.75] * 4
    labels = [1, 0, 0, 0, 1, 1, 1, 0]  # 1/4 right at 0.25, 3/4 right at 0.75
    assert expected_calibration_error(probs, labels, n_bins=4) == pytest.approx(0.0)


def test_an_overconfident_model_has_large_ece() -> None:
    # says 0.9 but is right only half the time: |0.9 - 0.5| = 0.4
    assert expected_calibration_error([0.9] * 4, [1, 1, 0, 0], n_bins=10) == pytest.approx(0.4)


def test_ece_weights_bins_by_size() -> None:
    probs = [
        0.9,
        0.9,
        0.9,
        0.1,
    ]  # bin 0.9: 3 items, acc 1/3 (gap 0.5667); bin 0.1: 1 item, acc 0 (gap 0.1)
    labels = [1, 0, 0, 0]
    expected = 3 / 4 * abs(0.9 - 1 / 3) + 1 / 4 * 0.1
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(expected)


def test_ece_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([], [])
