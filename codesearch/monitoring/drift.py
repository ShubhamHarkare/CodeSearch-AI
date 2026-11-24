"""Query drift: how far recent questions have moved from the golden set."""

import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from codesearch.interfaces import Embedder
from codesearch.monitoring.windows import RollingWindow

Vector = NDArray[np.float32]


def reference_questions(golden_path: Path) -> list[str]:
    path = Path(golden_path)
    if not path.exists():
        return []
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():  # skip blank lines
            row = json.loads(line)
            if row.get("expected") == "answer":
                questions.append(row["question"])
    return questions


def centroid(vectors: ArrayLike) -> Vector:
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim != 2 or len(v) == 0:
        raise ValueError("need a non-empty 2-D array")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # a zero row stays zero instead of becoming NaN
    out: Vector = (v / norms).mean(axis=0)  # (n, dim) -> (dim,)
    return out


def cosine_distance(a: ArrayLike, b: ArrayLike) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        raise ValueError("cosine distance is undefined for a zero vector")
    return float(1.0 - float(np.dot(a, b)) / (na * nb))


class DriftTracker:
    def __init__(self, embedder: Embedder, reference: list[str], min_questions: int = 20) -> None:
        if not reference:
            raise ValueError("need at least one reference question")

        self.reference = centroid(embedder.embed(reference))
        self.cache: dict[str, Vector] = {}  # trace_id -> embedding
        self.min_questions = min_questions
        self.embedder = embedder

    def distance(self, window: RollingWindow) -> float | None:
        items = window.snapshot()
        if len(items) < self.min_questions:
            return None

        new = [s for s in items if s.trace_id not in self.cache]
        if new:
            vectors = self.embedder.embed([s.query for s in new])  # ONE batched call
            for s, v in zip(new, vectors, strict=True):
                self.cache[s.trace_id] = v

        live = {s.trace_id for s in items}
        for trace_id in list(self.cache):  # list(): can't delete from a dict while iterating
            if trace_id not in live:
                del self.cache[trace_id]

        recent = centroid(np.stack([self.cache[s.trace_id] for s in items]))

        return cosine_distance(recent, self.reference)

    def cached(self) -> int:
        return len(self.cache)
