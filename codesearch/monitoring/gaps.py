"""Finds topics users ask about that the docs don't answer well.

What goes here:
    - ``GapCluster`` (dataclass): size, example_queries (top 5), centroid.
    - ``content_gaps(low_score_queries, embedder, k=5) -> list[GapCluster]``:
      k-means (scikit-learn) on low-score query embeddings, sorted by size.
    - Exposed at GET /v1/admin/gaps and shown on the dashboard.

Tested by: tests/unit/test_gaps.py with two obvious clusters.
Phase: 3
"""
