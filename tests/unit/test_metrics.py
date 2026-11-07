"""Tests for eval/metrics.py. Written BEFORE the code: they define what "correct" means.

Conventions the metrics must follow:
    - ``retrieved`` is de-duplicated first (first occurrence wins), because many
      chunks share one URL.
    - ``k <= 0`` and an empty ``relevant`` raise ValueError; an empty
      ``retrieved`` simply scores 0.0.
    - Ranks start at 1.
"""

import math

import pytest

from eval.metrics import (
    abstain_precision_recall,
    aggregate,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    to_page,
)

# A small shared example:  rank 1 = x, rank 2 = a, rank 3 = y, rank 4 = b
RETRIEVED = ["x", "a", "y", "b"]
RELEVANT = {"a", "b"}


# =====================================================================
# to_page
# =====================================================================


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://react.dev/reference/react/useState#usestate",
            "https://react.dev/reference/react/useState",
        ),
        ("https://react.dev/learn", "https://react.dev/learn"),
        ("https://react.dev/learn#", "https://react.dev/learn"),
        ("https://react.dev/", "https://react.dev/"),
    ],
)
def test_to_page(url: str, expected: str) -> None:
    assert to_page(url) == expected


# =====================================================================
# hit_rate_at_k
# =====================================================================


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1.0), (4, 1.0), (100, 1.0)])
def test_hit_rate_at_k(k: int, expected: float) -> None:
    assert hit_rate_at_k(RETRIEVED, RELEVANT, k) == expected


def test_hit_rate_with_no_hits() -> None:
    assert hit_rate_at_k(["x", "y", "z"], {"a"}, 3) == 0.0


# =====================================================================
# recall_at_k
# =====================================================================


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.5), (3, 0.5), (4, 1.0), (10, 1.0)])
def test_recall_at_k(k: int, expected: float) -> None:
    assert recall_at_k(RETRIEVED, RELEVANT, k) == expected


def test_recall_with_a_single_relevant_item_equals_hit_rate() -> None:
    for k in range(1, 5):
        assert recall_at_k(RETRIEVED, {"b"}, k) == hit_rate_at_k(RETRIEVED, {"b"}, k)


def test_relevant_items_never_retrieved_count_against_recall() -> None:
    assert recall_at_k(["a"], {"a", "b", "c", "d"}, 5) == 0.25


# =====================================================================
# mrr
# =====================================================================


@pytest.mark.parametrize(
    ("retrieved", "expected"),
    [
        (["a", "x", "y"], 1.0),
        (["x", "a", "y"], 0.5),
        (["x", "y", "a"], 1 / 3),
        (["x", "y", "z"], 0.0),
    ],
)
def test_mrr(retrieved: list[str], expected: float) -> None:
    assert mrr(retrieved, {"a"}) == pytest.approx(expected)


def test_mrr_uses_the_first_relevant_item_only() -> None:
    assert mrr(RETRIEVED, RELEVANT) == 0.5  # "a" at rank 2; "b" at rank 4 is ignored


def test_mrr_ignores_hits_beyond_k() -> None:
    assert mrr(["x", "y", "a"], {"a"}, k=2) == 0.0
    assert mrr(["x", "y", "a"], {"a"}, k=3) == pytest.approx(1 / 3)


def test_mrr_default_k_is_10() -> None:
    eleven = [f"n{i}" for i in range(10)] + ["a"]
    assert mrr(eleven, {"a"}) == 0.0


# =====================================================================
# ndcg_at_k
# =====================================================================


def test_ndcg_hand_computed_example() -> None:
    # hits at ranks 2 and 4: DCG = 1/log2(3) + 1/log2(5)
    # ideal: hits at ranks 1 and 2: IDCG = 1/log2(2) + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RETRIEVED, RELEVANT, k=4) == pytest.approx(dcg / idcg)  # ~0.6509


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["a", "b", "x"], RELEVANT, k=3) == pytest.approx(1.0)


def test_ndcg_with_no_hits_is_zero() -> None:
    assert ndcg_at_k(["x", "y"], RELEVANT, k=2) == 0.0


def test_ndcg_ideal_is_capped_at_k() -> None:
    # 3 relevant items but k=1: the best possible is ONE hit at rank 1,
    # so a hit at rank 1 is perfect.
    assert ndcg_at_k(["a", "x"], {"a", "b", "c"}, k=1) == pytest.approx(1.0)


def test_ndcg_rewards_earlier_hits() -> None:
    assert ndcg_at_k(["a", "x", "y"], {"a"}, k=3) > ndcg_at_k(["x", "y", "a"], {"a"}, k=3)


def test_ndcg_stays_between_zero_and_one() -> None:
    for retrieved in (["a", "b"], ["b", "x", "a"], ["x"], ["x", "y", "z", "a"]):
        assert 0.0 <= ndcg_at_k(retrieved, RELEVANT, k=3) <= 1.0


# =====================================================================
# Duplicates and other shared rules (apply to every ranking metric)
# =====================================================================

RANKING_METRICS = [hit_rate_at_k, recall_at_k, mrr, ndcg_at_k]


@pytest.mark.parametrize("metric", RANKING_METRICS, ids=lambda f: f.__name__)
def test_duplicates_are_removed_before_ranking(metric) -> None:  # type: ignore[no-untyped-def]
    # Three chunks from page "x" must not push "a" down to rank 4.
    with_duplicates = ["x", "x", "x", "a"]
    without = ["x", "a"]
    assert metric(with_duplicates, {"a"}, 2) == metric(without, {"a"}, 2)


def test_duplicates_cannot_push_recall_above_one() -> None:
    assert recall_at_k(["a", "a", "a"], {"a"}, 3) == 1.0


@pytest.mark.parametrize("metric", RANKING_METRICS, ids=lambda f: f.__name__)
def test_empty_retrieved_scores_zero(metric) -> None:  # type: ignore[no-untyped-def]
    assert metric([], {"a"}, 5) == 0.0


@pytest.mark.parametrize("metric", RANKING_METRICS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises(metric, k: int) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        metric(RETRIEVED, RELEVANT, k)


@pytest.mark.parametrize("metric", RANKING_METRICS, ids=lambda f: f.__name__)
def test_empty_relevant_raises(metric) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        metric(RETRIEVED, set(), 5)


@pytest.mark.parametrize("metric", RANKING_METRICS, ids=lambda f: f.__name__)
def test_relevant_can_be_any_collection(metric) -> None:  # type: ignore[no-untyped-def]
    assert metric(RETRIEVED, ["a", "b"], 4) == metric(RETRIEVED, {"a", "b"}, 4)


@pytest.mark.parametrize("metric", RANKING_METRICS, ids=lambda f: f.__name__)
def test_inputs_are_not_modified(metric) -> None:  # type: ignore[no-untyped-def]
    retrieved = ["x", "x", "a"]
    relevant = {"a"}
    metric(retrieved, relevant, 3)
    assert retrieved == ["x", "x", "a"]
    assert relevant == {"a"}


def test_page_level_scoring_by_mapping_through_to_page() -> None:
    """Section URLs miss, but the same pages hit once both sides go through to_page."""
    retrieved = [
        "https://react.dev/reference/react/useState#usestate",
        "https://react.dev/learn#components",
    ]
    relevant = {"https://react.dev/reference/react/useState#updating-state"}

    assert hit_rate_at_k(retrieved, relevant, 2) == 0.0  # section level: miss
    pages = [to_page(u) for u in retrieved]
    assert hit_rate_at_k(pages, {to_page(u) for u in relevant}, 2) == 1.0  # page level: hit


# =====================================================================
# abstain_precision_recall
# =====================================================================


def test_abstain_precision_and_recall() -> None:
    # True = abstained (predicted) / should abstain (expected)
    predicted = [True, True, False, False, True]
    expected = [True, False, False, True, True]
    # true positives = 2 (rows 0 and 4); predicted positives = 3; expected positives = 3
    assert abstain_precision_recall(predicted, expected) == pytest.approx((2 / 3, 2 / 3))


def test_abstain_perfect() -> None:
    assert abstain_precision_recall([True, False], [True, False]) == (1.0, 1.0)


def test_abstain_never_predicted_gives_zero_precision() -> None:
    assert abstain_precision_recall([False, False], [True, False]) == (0.0, 0.0)


def test_abstain_nothing_expected_gives_zero_recall() -> None:
    assert abstain_precision_recall([True, False], [False, False]) == (0.0, 0.0)


def test_abstain_empty_input() -> None:
    assert abstain_precision_recall([], []) == (0.0, 0.0)


def test_abstain_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        abstain_precision_recall([True], [True, False])


# =====================================================================
# aggregate
# =====================================================================


def test_aggregate_means_each_metric() -> None:
    rows = [{"recall@5": 1.0, "mrr": 0.5}, {"recall@5": 0.0, "mrr": 1.0}]
    assert aggregate(rows) == pytest.approx({"recall@5": 0.5, "mrr": 0.75})


def test_aggregate_only_averages_rows_that_have_the_metric() -> None:
    # Out-of-scope rows have no recall; they must not drag recall's mean down.
    rows = [{"recall@5": 1.0}, {"recall@5": 0.5}, {"abstained": 1.0}]
    assert aggregate(rows) == pytest.approx({"recall@5": 0.75, "abstained": 1.0})


def test_aggregate_of_nothing_is_empty() -> None:
    assert aggregate([]) == {}
