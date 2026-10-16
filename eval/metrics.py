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
