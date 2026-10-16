"""Operations endpoints.

Endpoints:
    - ``GET /healthz``: process is up (always 200 if the app runs).
    - ``GET /readyz``: index loaded + Redis reachable (or "degraded" with the
      in-memory fallback); 503 when the index isn't ready.
    - ``GET /v1/index``: manifest (index_version, corpus SHA, chunk count,
      last update, changelog tail), prompt version, image SHA.
    - ``POST /v1/admin/reindex``: HMAC-verified (deps.verify_webhook);
      schedules build_index in the background and returns 202 immediately.
    - ``GET /v1/admin/gaps``: content-gap clusters.
    - ``GET /metrics``: Prometheus exposition.

Tested by: tests/integration/test_api.py.
Phase: 3-4
"""
