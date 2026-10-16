"""Builds the FastAPI application.

What goes here:
    - ``create_app(settings: Settings | None = None) -> FastAPI``:
        * ``lifespan``: configure logging; build the pipeline via factory.py;
          load the index; start background tasks (api/background.py); stop
          them cleanly on shutdown
        * middleware: create / accept ``X-Trace-Id``, bind it to logs, return
          it in the response header; record latency
        * exception handler: CodeSearchError -> JSON {error, trace_id} with its
          http_status; anything else -> 500 with a generic message (and full
          traceback in logs only)
        * include routers from routes_ask.py and routes_admin.py

Must not:
    - contain endpoint logic.

Tested by: tests/integration/test_api.py using httpx.AsyncClient against
create_app(fake settings).
Phase: 1
"""
