"""FastAPI dependency providers.

What goes here:
    - ``get_settings(request)``, ``get_pipeline(request)``,
      ``get_trace_store(request)``, ``get_feedback_store(request)``,
      ``get_index_handle(request)``: each reads from ``request.app.state``.
    - ``verify_webhook(request, settings)``: HMAC-SHA256 check of the
      ``X-Hub-Signature-256`` header for the reindex webhook; raises
      InvalidWebhookError. Use ``hmac.compare_digest``.

Tested by: tests/integration/test_api.py (bad signature -> 401).
Phase: 1, 3 (webhook)
"""
