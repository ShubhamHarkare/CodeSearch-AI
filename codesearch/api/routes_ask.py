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

from fastapi import APIRouter, Request

from codesearch.api.deps import PipelineDep
from codesearch.api.models import AskRequest, AskResponse

router = APIRouter(prefix="/v1", tags=["ask"])


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request, pipeline: PipelineDep) -> AskResponse:
    # 1. answer = await pipeline.answer(question, history, bypass_cache=..., trace_id=...)
    answer = await pipeline.answer(
        question=body.question,
        history=body.history,
        bypass_cache=body.bypass_cache,
        trace_id=request.state.trace_id,
    )

    return AskResponse.from_answer(answer)
