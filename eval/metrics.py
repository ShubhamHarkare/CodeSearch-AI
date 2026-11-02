"""Scoring functions for evaluation. Pure, fully unit-tested.

What goes here:
    - ``recall_at_k(retrieved_urls, expected_urls, k) -> float`` (match on URL;
      decide and document whether the #anchor must match too)
    - ``mrr(retrieved_urls, expected_urls, k=10) -> float``
    - ``ndcg_at_k(retrieved_urls, expected_urls, k=10) -> float``
    - ``abstain_precision_recall(predictions, labels) -> tuple[float, float]``
    - ``citation_precision(sentence_supports) -> float``: share of cited
      sentences whose cited chunk supports them
    - ``aggregate(rows) -> dict``: means, plus per-category breakdown

Tested by: tests/unit/test_metrics.py with hand-computed examples.
Phase: 1 (retrieval metrics), 2 (citation precision)
"""

import math
from collections.abc import Collection, Mapping, Sequence


def to_page(url: str) -> str:
    """Drop the #anchor so section URLs compare at page level."""
    return url.split("#")[0]


def _prepare(reterived: Sequence[str], relevant: Collection[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if not relevant:
        raise ValueError("relevant must not be empty")

    unique_reterived = list(dict.fromkeys(reterived).keys())[:k]

    return unique_reterived


def hit_rate_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 if any relevant item is in the top k, else 0.0."""
    top_k = _prepare(retrieved, relevant, k)
    relevant_set = set(relevant)

    for item in top_k:
        if item in relevant_set:
            return 1.0
    return 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    top_k = _prepare(reterived=retrieved, relevant=relevant, k=k)
    count = 0
    relevant_set = set(relevant)

    for item in top_k:
        if item in relevant_set:
            count += 1

    return count / len(relevant_set)


def mrr(retrieved: Sequence[str], relevant: Collection[str], k: int = 10) -> float:
    top_k = _prepare(retrieved, relevant, k)

    relevant_set = set(relevant)

    for rank, item in enumerate(top_k, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int = 10) -> float:
    top_k = _prepare(retrieved, relevant, k)
    relevant_set = set(relevant)
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant_set:
            dcg += 1 / math.log2(rank + 1)

    n_ideal = min(len(relevant_set), k)
    idcg = 0.0
    for i in range(1, n_ideal + 1):
        idcg += 1 / math.log2(i + 1)

    return dcg / idcg


def abstain_precision_recall(
    predicted: Sequence[bool], expected: Sequence[bool]
) -> tuple[float, float]:
    """Precision and recall for abstaining, where True means "abstain".

    predicted[i]: the system abstained on question i.
    expected[i]:  question i is out of scope and should be abstained on.
    Returns (precision, recall); a 0/0 ratio is 0.0.
    """
    if len(predicted) != len(expected):
        raise ValueError(
            f"predicted ({len(predicted)}) and expected ({len(expected)}) must have the same length"
        )

    tp = fp = fn = 0
    for p, e in zip(predicted, expected, strict=True):
        if p and e:
            tp += 1
        elif p and not e:
            fp += 1
        elif not p and e:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    values_per_metric: dict[str, float] = {}
    count_per_metric: dict[str, int] = {}

    for row in rows:
        for key, value in row.items():
            values_per_metric[key] = values_per_metric.get(key, 0.0) + value
            count_per_metric[key] = count_per_metric.get(key, 0) + 1

    aggregate_dict: dict[str, float] = {}

    for key in values_per_metric:
        aggregate_dict[key] = values_per_metric[key] / count_per_metric[key]

    return aggregate_dict
