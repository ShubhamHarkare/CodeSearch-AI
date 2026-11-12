"""Request and response bodies for the HTTP API.

Kept separate from schemas.py so the public API contract can change without
touching internals (and vice versa).

What goes here:
    - ``AskRequest``: question (1..1000 chars), history (list of {role, content},
      max 10), bypass_cache.
    - ``AskResponse``: answer, citations, abstained, confidence, groundedness,
      cache_tier, timings, trace_id.
    - ``FeedbackRequest``: trace_id, vote (+1 / -1), comment (optional, max 500).
    - ``IndexInfo``, ``HealthStatus`` (status, components, versions, image_sha).

Phase: 1
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from codesearch.ingest.manifest import ChangeLogEntry
from codesearch.schemas import Answer, ChatMessage, Citation


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=1000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=10)
    bypass_cache: bool = False


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    confidence: float | None = None
    groundedness: float | None = None
    cache_tier: Literal["exact", "semantic"] | None = None
    timings: dict[str, float]
    trace_id: str
    index_version: str
    prompt_version: str

    @classmethod
    def from_answer(cls, a: Answer) -> "AskResponse":
        return cls(
            answer=a.text,
            citations=a.citations,
            abstained=a.abstained,
            confidence=a.confidence,
            groundedness=a.groundedness,
            cache_tier=a.cache_tier,
            timings=a.timings,
            trace_id=a.trace_id,
            index_version=a.index_version,
            prompt_version=a.prompt_version,
        )


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    components: dict[str, str]  # {"index": "ok"|"missing", "cache": "ok"|"degraded"|"disabled"}
    versions: dict[str, str]  # prompt_version always; index_version only if ready
    image_sha: str | None = None


class IndexInfo(BaseModel):
    index_version: str
    corpus_sha: str
    embedding_model: str
    chunk_count: int
    built_at: datetime
    changelog: list[ChangeLogEntry]
    prompt_version: str
    image_sha: str | None = None
