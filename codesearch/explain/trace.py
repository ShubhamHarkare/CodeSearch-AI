"""Records how each answer was produced and stores it by trace_id.

What goes here:
    - ``TraceBuilder``: the pipeline calls ``add_candidates()``,
      ``set_used()``, ``set_sentences()``, ``set_abstain()``, ``time(step)``
      (context manager) and finally ``build() -> ExplanationTrace``.
    - ``TraceStore(db_path)``: SQLite; ``save(trace)``, ``get(trace_id)``
      (raises TraceNotFoundError), ``recent(n)``; stores the trace as JSON.
    - per-source "why" data: BM25 term contributions (index/bm25.py) attached
      to each used chunk.

Served by: GET /v1/explain/{trace_id}; rendered in the UI's
"Why this answer?" panel.

Tested by: tests/unit/test_trace.py (round-trip; timings recorded).
Phase: 2
"""

import sqlite3
import threading
import time as _time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from codesearch.errors import TraceNotFoundError
from codesearch.schemas import ExplanationTrace, ScoredChunk, SentenceSupport


class TraceBuilder:
    def __init__(self, trace_id: str, question: str) -> None:
        self.trace_id = trace_id
        self.question = question
        self.standalone_question = question  # same unless there was history
        self._candidates: list[ScoredChunk] = []
        self._used: list[str] = []
        self._sentences: list[SentenceSupport] = []
        self._abstain_reason: str | None = None
        self._cache_tier: Literal["exact", "semantic"] | None = None
        self._versions: dict[str, str] = {}
        self.timings: dict[str, float] = {}
        self._matched_terms: dict[str, dict[str, float]] = {}

    def set_standalone(self, question: str) -> None:
        self.standalone_question = question

    def add_candidates(self, chunks: Sequence[ScoredChunk]) -> None:
        self._candidates.extend(chunks)

    def set_used(self, chunks: Sequence[ScoredChunk]) -> None:
        self._used = [c.chunk.id for c in chunks]
        position = {c.chunk.id: i for i, c in enumerate(self._candidates)}

        for c in chunks:
            if c.chunk.id in position:
                self._candidates[position[c.chunk.id]] = c
            else:
                self._candidates.append(c)

    def set_used_ids(self, ids: Sequence[str]) -> None:
        self._used = list(ids)

    def set_sentences(self, sentences: Sequence[SentenceSupport]) -> None:
        self._sentences = list(sentences)

    def set_abstain(self, reason: str) -> None:
        self._abstain_reason = reason

    def set_cache_tier(self, tier: Literal["exact", "semantic"]) -> None:
        self._cache_tier = tier

    def set_versions(self, **versions: str) -> None:
        self._versions.update(versions)

    def set_matched_terms(self, terms: dict[str, dict[str, float]]) -> None:
        self._matched_terms = dict(terms)

    @contextmanager
    def time(self, step: str) -> Iterator[None]:
        start = _time.perf_counter()
        try:
            yield
        finally:
            self.timings[f"{step}_ms"] = (_time.perf_counter() - start) * 1000.0

    def build(self) -> ExplanationTrace:
        return ExplanationTrace(
            trace_id=self.trace_id,
            question=self.question,
            standalone_question=self.standalone_question,
            candidates=list(self._candidates),
            used_chunk_ids=list(self._used),
            sentences=list(self._sentences),
            abstain_reason=self._abstain_reason,
            timings=dict(self.timings),
            versions=dict(self._versions),
            cache_tier=self._cache_tier,
            matched_terms=self._matched_terms,
        )


class TraceStore:
    """Traces in SQLite, stored as JSON, keyed by trace_id."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # FastAPI may call us from different threads: allow that, and use a lock instead.
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS traces ("
                "trace_id TEXT PRIMARY KEY, created_at REAL NOT NULL, data TEXT NOT NULL)"
            )

    def save(self, trace: ExplanationTrace) -> None:
        with self._lock, self._db:  # `with self._db` commits (or rolls back on error)
            self._db.execute(
                "INSERT OR REPLACE INTO traces (trace_id, created_at, data) VALUES (?, ?, ?)",
                (trace.trace_id, _time.time(), trace.model_dump_json()),
            )

    def get(self, trace_id: str) -> ExplanationTrace:
        with self._lock:
            row = self._db.execute(
                "SELECT data FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        if row is None:
            raise TraceNotFoundError(f"no trace {trace_id!r}")
        return ExplanationTrace.model_validate_json(row[0])

    def recent(self, n: int = 20) -> list[ExplanationTrace]:
        with self._lock:
            rows = self._db.execute(
                "SELECT data FROM traces ORDER BY rowid DESC LIMIT ?", (n,)
            ).fetchall()
        return [ExplanationTrace.model_validate_json(data) for (data,) in rows]

    def __len__(self) -> int:
        with self._lock:
            (count,) = self._db.execute("SELECT COUNT(*) FROM traces").fetchone()
        return int(count)

    def close(self) -> None:
        self._db.close()
