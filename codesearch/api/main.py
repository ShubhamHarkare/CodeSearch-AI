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

import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from codesearch.api import routes_admin, routes_ask
from codesearch.config import Settings
from codesearch.errors import CodeSearchError
from codesearch.factory import build_pipeline
from codesearch.log import bind_trace_id, configure_logging, get_logger, new_trace_id
from codesearch.pipeline import AnswerPipeline

TRACE_HEADER = "X-Trace-Id"
_SAFE_TRACE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
log = get_logger(__name__)


def create_app(
    settings: Settings | None = None, *, pipeline: AnswerPipeline | None = None
) -> FastAPI:
    settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.settings = settings
        app.state.pipeline = pipeline if pipeline is not None else build_pipeline(settings)
        log.info("Application startup complete")
        # STARTUP (runs once, before the first request):
        #   configure_logging()
        #   app.state.settings = settings
        #   app.state.pipeline = pipeline if given, else build_pipeline(settings)
        yield
        # SHUTDOWN (runs once when the server stops): just log for now
        log.info("Application shutdown complete")

    app = FastAPI(title="CodeSearch AI", version="2.0", lifespan=lifespan)

    @app.middleware("http")
    async def trace(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # 1. incoming = request.headers.get(TRACE_HEADER, "")
        incoming = request.headers.get(TRACE_HEADER, "")
        #    trace_id = incoming if _SAFE_TRACE_ID.fullmatch(incoming) else new_trace_id()
        trace_id = incoming if _SAFE_TRACE_ID.fullmatch(incoming) else new_trace_id()
        # 2. request.state.trace_id = trace_id; bind_trace_id(trace_id)
        request.state.trace_id = trace_id
        bind_trace_id(trace_id)
        # 3. start = time.perf_counter(); response = await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        # 4. response.headers[TRACE_HEADER] = trace_id
        response.headers[TRACE_HEADER] = trace_id
        # 5. log.info("http.request", method=..., path=request.url.path,
        #             status=response.status_code, latency_ms=...)
        log.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        # 6. return response
        return response

    @app.exception_handler(CodeSearchError)
    async def known_error(request: Request, exc: CodeSearchError) -> JSONResponse:
        # trace_id = getattr(request.state, "trace_id", None) or new_trace_id()
        trace_id = getattr(request.state, "trace_id", None) or new_trace_id()
        log.warning("http.error", error=type(exc).__name__, detail=str(exc))
        # log.warning("http.error", error=type(exc).__name__, detail=str(exc))
        return JSONResponse(
            {"error": exc.public_message, "trace_id": trace_id},
            status_code=exc.http_status,
            headers={TRACE_HEADER: trace_id},
        )

    @app.exception_handler(Exception)
    async def unknown_error(request: Request, exc: Exception) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None) or new_trace_id()
        log.exception("http.unhandled_error", error=type(exc).__name__, detail=str(exc))
        public_msg = getattr(CodeSearchError, "public_message", "An unexpected error occurred.")
        return JSONResponse(
            content={"error": public_msg, "trace_id": trace_id},
            status_code=500,
            headers={TRACE_HEADER: trace_id},
        )

    app.include_router(routes_ask.router)
    app.include_router(routes_admin.router)
    return app
