"""One exception hierarchy for the whole package.

What goes here:
    - ``CodeSearchError(Exception)``: base class with a safe, user-facing
      ``public_message`` and an ``http_status``.
    - Subclasses: ``ConfigError``, ``CorpusFetchError``, ``IndexNotReady``
      (503), ``LLMUnavailable`` (503), ``LLMTimeout`` (504),
      ``InvalidWebhook`` (401), ``TraceNotFound`` (404).

What it must do:
    - Let api/main.py map any CodeSearchError to an HTTP response without
      leaking internal error text to users.

Must not:
    - be bypassed: raise these, never bare ``Exception``; catch broad
      exceptions only at the API boundary.

Phase: 0
"""
