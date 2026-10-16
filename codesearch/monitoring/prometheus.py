"""Prometheus metrics. The only module that imports prometheus_client.

What goes here:
    - Histograms: ``codesearch_ttft_seconds``, ``codesearch_latency_seconds``
      (label: cache_tier), ``codesearch_retrieval_seconds``.
    - Counters: ``codesearch_requests_total`` (labels: abstained, cache_tier,
      status), ``codesearch_reindex_total``.
    - Gauges: rolling ``groundedness_median``, ``abstain_rate``,
      ``drift_distance``, ``cache_hit_rate``, ``index_chunks``,
      ``index_info`` (labels: index_version, corpus_sha).
    - ``observe(signals)`` per request and ``refresh(window)`` for the gauges.

Served by: GET /metrics (api/routes_admin.py). Grafana alert rules live in
deploy/grafana/.

Tested by: tests/unit/test_prometheus.py (observe increments; refresh sets gauges).
Phase: 3
"""
