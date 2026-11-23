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

import hashlib
import hmac
from typing import Annotated

from fastapi import Depends, Request

from codesearch.api.background import ReindexRunner
from codesearch.config import Settings
from codesearch.errors import InvalidWebhookError, TraceNotFoundError
from codesearch.explain.trace import TraceStore
from codesearch.index.handle import IndexHandle
from codesearch.monitoring.feedback import FeedbackStore
from codesearch.monitoring.prometheus import Monitor
from codesearch.pipeline import AnswerPipeline

SIGNATURE_HEADER = "X-Hub-Signature-256"  # how GitHub signs webhook deliveries


def get_settings(request: Request) -> Settings:
    # return request.app.state.settings
    settings: Settings = request.app.state.settings
    return settings


def get_pipeline(request: Request) -> AnswerPipeline:
    # return request.app.state.pipeline
    pipeline: AnswerPipeline = request.app.state.pipeline
    return pipeline


def get_index_handle(request: Request) -> IndexHandle:
    # the pipeline already holds the handle: get_pipeline(request).handle
    handle: IndexHandle = get_pipeline(request).handle
    return handle


def get_trace_store(request: Request) -> TraceStore:
    store: TraceStore | None = get_pipeline(request).trace_store
    if store is None:
        raise TraceNotFoundError("trace store disabled")
    return store


def get_reindexer(request: Request) -> ReindexRunner:
    runner: ReindexRunner = request.app.state.reindexer
    return runner


def get_monitor(request: Request) -> Monitor:
    monitor: Monitor = request.app.state.monitor
    return monitor


def get_feedback_store(request: Request) -> FeedbackStore:
    feedback: FeedbackStore = request.app.state.feedback
    return feedback


def check_signature(secret: str, body: bytes, header: str | None) -> None:
    """Raise InvalidWebhookError unless header == "sha256=" + HMAC-SHA256(secret, body).

    Only someone who knows the secret can produce a valid signature for a body, so
    only they can trigger a rebuild. compare_digest takes the same time however many
    characters match, so the signature can't be guessed by timing responses.
    """
    if not secret:  # "" means not configured: nothing can be verified, so refuse
        raise InvalidWebhookError(
            "no webhook secret configured",
            public_message="Re-indexing by webhook is not enabled.",
        )
    if not header or not header.startswith("sha256="):
        raise InvalidWebhookError("missing or malformed signature header")

    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, header):
        raise InvalidWebhookError("signature mismatch")


async def verify_webhook(request: Request) -> None:
    """Dependency for POST /v1/admin/reindex: a bad signature becomes a 401."""
    settings = get_settings(request)
    check_signature(
        settings.admin.webhook_secret.get_secret_value(),
        await request.body(),
        request.headers.get(SIGNATURE_HEADER),
    )


SettingsDep = Annotated[Settings, Depends(get_settings)]
PipelineDep = Annotated[AnswerPipeline, Depends(get_pipeline)]
HandleDep = Annotated[IndexHandle, Depends(get_index_handle)]
TraceStoreDep = Annotated[TraceStore, Depends(get_trace_store)]
ReindexerDep = Annotated[ReindexRunner, Depends(get_reindexer)]
MonitorDep = Annotated[Monitor, Depends(get_monitor)]
FeedbackStoreDep = Annotated[FeedbackStore, Depends(get_feedback_store)]
