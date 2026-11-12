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

from typing import Annotated

from fastapi import Depends, Request

from codesearch.config import Settings
from codesearch.index.handle import IndexHandle
from codesearch.pipeline import AnswerPipeline


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


SettingsDep = Annotated[Settings, Depends(get_settings)]
PipelineDep = Annotated[AnswerPipeline, Depends(get_pipeline)]
HandleDep = Annotated[IndexHandle, Depends(get_index_handle)]
