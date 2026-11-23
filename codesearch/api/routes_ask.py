"""User-facing endpoints. Each handler: validate, call the pipeline, shape the response.
Endpoints:
    - ``POST /v1/ask`` -> AskResponse (JSON).
    - ``POST /v1/ask/stream`` -> text/event-stream; events in this order:
      retrieval (sources + scores), token (repeated), citations, groundedness,
      done (timings, trace_id); error if something fails mid-stream.
    - ``GET /v1/explain/{trace_id}`` -> the stored ExplanationTrace.
    - ``POST /v1/explain/{trace_id}/deep`` -> leave-one-out attribution
      (rate-limit it: it costs k LLM calls).
    - ``POST /v1/feedback`` -> 204.

Must not:
    - contain retrieval or LLM logic.

Tested by: tests/integration/test_api.py (JSON answer, SSE event order,
404 on unknown trace, feedback stored).
Phase: 1 (ask), 2 (explain), 3 (stream, feedback)
"""

from collections.abc import AsyncGenerator
from contextlib import aclosing

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from codesearch.api.deps import FeedbackStoreDep, MonitorDep, PipelineDep, TraceStoreDep
from codesearch.api.models import AskRequest, AskResponse, DeepExplainResponse, FeedbackRequest
from codesearch.api.sse import stream_response
from codesearch.monitoring.prometheus import Monitor
from codesearch.schemas import Answer, ExplanationTrace, StreamEvent

router = APIRouter(prefix="/v1", tags=["ask"])


@router.get("/explain/{trace_id}", response_model=ExplanationTrace)
async def explain(trace_id: str, store: TraceStoreDep) -> ExplanationTrace:
    return store.get(trace_id)


@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest, request: Request, pipeline: PipelineDep, monitor: MonitorDep
) -> AskResponse:
    # 1. answer = await pipeline.answer(question, history, bypass_cache=..., trace_id=...)
    answer = await pipeline.answer(
        question=body.question,
        history=body.history,
        bypass_cache=body.bypass_cache,
        trace_id=request.state.trace_id,
    )
    monitor.record(answer, body.question)  # counters, histograms and the rolling window
    return AskResponse.from_answer(answer)


@router.post("/explain/{trace_id}/deep", response_model=DeepExplainResponse)
async def deep_explain(trace_id: str, pipeline: PipelineDep) -> DeepExplainResponse:
    """Re-asks the question once per source with that source removed (slow: k LLM calls)."""
    scores = await pipeline.deep_explain(trace_id)
    return DeepExplainResponse(trace_id=trace_id, scores=scores)


async def recorded(
    events: AsyncGenerator[StreamEvent], monitor: Monitor, question: str
) -> AsyncGenerator[StreamEvent]:
    """Pass the pipeline's events through; record the answer when "done" goes by.

    aclosing: if the client leaves, closing this generator also closes the pipeline
    under it (and so the LLM stream), exactly as without this extra layer.
    """
    async with aclosing(events) as source:
        async for event in source:
            if event.type == "done":
                monitor.record(Answer.model_validate(event.data), question)
            yield event


@router.post("/ask/stream")
async def ask_stream(
    body: AskRequest, request: Request, pipeline: PipelineDep, monitor: MonitorDep
) -> StreamingResponse:
    """Stream the answer as server-sent events: retrieval, token..., done (or error)."""
    trace_id = request.state.trace_id
    events = pipeline.stream(
        question=body.question,
        history=body.history,
        bypass_cache=body.bypass_cache,
        trace_id=trace_id,
    )
    return StreamingResponse(
        stream_response(recorded(events, monitor, body.question), request, trace_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/feedback", status_code=204)
async def feedback(
    body: FeedbackRequest, store: FeedbackStoreDep, traces: TraceStoreDep, monitor: MonitorDep
) -> Response:
    traces.get(body.trace_id)  # unknown id -> TraceNotFoundError -> 404, nothing stored
    store.add(body.trace_id, body.vote, body.comment)
    monitor.metrics.observe_feedback(body.vote)
    return Response(status_code=204)
