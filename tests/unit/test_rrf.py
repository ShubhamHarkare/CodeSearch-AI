"""Tests for rrf_fuse in codesearch/retrieval/hybrid.py (Reciprocal Rank Fusion).

score(d) = sum over every ranked list that contains d of 1 / (k + rank_d),
with ranks starting at 1. Pure function: no index, no embedder.
"""

from __future__ import annotations

import copy

import pytest

from codesearch.retrieval.hybrid import rrf_fuse


def ids(fused: list[tuple[str, float]]) -> list[str]:
    return [doc for doc, _ in fused]


def scores(fused: list[tuple[str, float]]) -> dict[str, float]:
    return dict(fused)


# =====================================================================
# The formula
# =====================================================================


def test_hand_computed_example() -> None:
    fused = rrf_fuse({"dense": ["a", "b", "c"], "bm25": ["c", "a", "d"]}, k=60)
    s = scores(fused)
    assert s["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert s["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert s["b"] == pytest.approx(1 / 62)
    assert s["d"] == pytest.approx(1 / 63)
    assert ids(fused) == ["a", "c", "b", "d"]


def test_default_k_is_60() -> None:
    assert rrf_fuse({"x": ["a"]}) == [("a", pytest.approx(1 / 61))]


def test_ranks_start_at_one() -> None:
    """With k=0 the top item of a list scores exactly 1/1."""
    assert scores(rrf_fuse({"x": ["a", "b"]}, k=0)) == pytest.approx({"a": 1.0, "b": 0.5})


def test_item_in_both_lists_beats_items_in_one() -> None:
    fused = rrf_fuse({"dense": ["a", "b"], "bm25": ["b", "c"]}, k=0)
    # a = 1, b = 1/2 + 1 = 1.5, c = 1/2
    assert ids(fused) == ["b", "a", "c"]
    assert scores(fused)["b"] == pytest.approx(1.5)


def test_item_in_only_one_list_still_scores() -> None:
    s = scores(rrf_fuse({"dense": ["a"], "bm25": ["b"]}, k=60))
    assert s == pytest.approx({"a": 1 / 61, "b": 1 / 61})


def test_works_with_more_than_two_lists() -> None:
    fused = rrf_fuse({"one": ["a", "b"], "two": ["b", "a"], "three": ["b"]}, k=60)
    assert ids(fused) == ["b", "a"]
    assert scores(fused)["b"] == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)


def test_list_names_do_not_change_the_result() -> None:
    a = rrf_fuse({"dense": ["a", "b"], "bm25": ["c"]})
    b = rrf_fuse({"left": ["a", "b"], "right": ["c"]})
    assert a == b


def test_larger_k_flattens_the_rank_differences() -> None:
    small = scores(rrf_fuse({"x": ["a", "b"]}, k=1))
    large = scores(rrf_fuse({"x": ["a", "b"]}, k=1000))
    assert small["a"] / small["b"] > large["a"] / large["b"]


# =====================================================================
# Ordering and ties
# =====================================================================


def test_result_is_sorted_by_score_descending() -> None:
    fused = rrf_fuse({"x": list("abcdef"), "y": list("fedcba")}, k=5)
    values = [score for _, score in fused]
    assert values == sorted(values, reverse=True)


def test_ties_keep_first_appearance_order() -> None:
    """Equal scores: the item seen first (lists in dict order, then rank) comes first."""
    assert ids(rrf_fuse({"dense": ["a", "b"], "bm25": ["b", "a"]})) == ["a", "b"]
    assert ids(rrf_fuse({"bm25": ["b", "a"], "dense": ["a", "b"]})) == ["b", "a"]


def test_ties_across_lists_are_deterministic() -> None:
    runs = {tuple(ids(rrf_fuse({"d": ["a", "x"], "b": ["y", "z"]}))) for _ in range(5)}
    assert runs == {("a", "y", "x", "z")}


# =====================================================================
# Edge cases
# =====================================================================


def test_every_item_appears_once() -> None:
    fused = rrf_fuse({"dense": ["a", "b", "c"], "bm25": ["c", "b", "a"]})
    assert sorted(ids(fused)) == ["a", "b", "c"]


def test_duplicates_inside_one_list_count_once_at_their_first_rank() -> None:
    """A repeated id must not score twice, and must not push later ranks down."""
    s = scores(rrf_fuse({"x": ["a", "a", "b"]}, k=0))
    assert s == pytest.approx({"a": 1.0, "b": 0.5})


def test_empty_rankings_give_empty_result() -> None:
    assert rrf_fuse({}) == []


def test_empty_lists_are_ignored() -> None:
    assert rrf_fuse({"dense": [], "bm25": ["a"]}, k=60) == [("a", pytest.approx(1 / 61))]
    assert rrf_fuse({"dense": [], "bm25": []}) == []


def test_k_zero_is_allowed() -> None:
    assert rrf_fuse({"x": ["a"]}, k=0) == [("a", pytest.approx(1.0))]


@pytest.mark.parametrize("k", [-1, -60])
def test_negative_k_is_rejected(k: int) -> None:
    with pytest.raises(ValueError):
        rrf_fuse({"x": ["a"]}, k=k)


def test_inputs_are_not_mutated() -> None:
    rankings = {"dense": ["a", "b", "a"], "bm25": ["c"]}
    before = copy.deepcopy(rankings)
    rrf_fuse(rankings)
    assert rankings == before


def test_scores_are_plain_floats() -> None:
    fused = rrf_fuse({"x": ["a", "b"]})
    assert all(type(score) is float for _, score in fused)
