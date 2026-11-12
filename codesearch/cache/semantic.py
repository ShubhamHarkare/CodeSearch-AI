"""Near-duplicate question tier: reuse an answer for a paraphrased question.

What goes here:
    - ``SemanticCache(embedder, threshold=0.95, max_entries=2000)``:
        * ``lookup(query, index_version, prompt_version) -> Answer | None``:
          embed the query; brute-force cosine against stored query vectors
          (numpy matrix, fine at this size) filtered to the same versions
        * a hit needs cosine >= threshold AND the same top-1 source as a fresh
          retrieval would give (pass that in), to avoid "useState" vs
          "useReducer" false hits
        * ``add(query, vector, answer)``; evict oldest beyond max_entries
    - Optionally persist vectors in Redis; in-process is fine to start.

Tested by: tests/unit/test_semantic_cache.py with HashEmbedder (exact
paraphrase hits; different topic misses; version mismatch misses).
Phase: 1
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from codesearch.cache.keys import normalize
from codesearch.interfaces import Embedder
from codesearch.schemas import Answer


@dataclass
class _Entry:  # the dataclass: what ONE cached question looks like
    question: str  # normalized question, e.g. "how do i use usestate"
    versions: tuple[str, str, str]  # (index_version, prompt_version, model)
    top1: str  # id of the best retrieved chunk
    vector: NDArray[np.float32]  # unit-length embedding of the question
    answer: Answer  # the answer to reuse


class SemanticCache:
    def __init__(
        self, embedder: Embedder, threshold: float = 0.95, max_entries: int = 1000
    ) -> None:
        if not (0 < threshold <= 1):
            raise ValueError

        if max_entries <= 0:
            raise ValueError

        self.embedder = embedder
        self.threshold = threshold
        self.max_entries = max_entries
        self._entries: list[_Entry] = []

    @staticmethod
    def _unit(v: NDArray[np.float32]) -> NDArray[np.float32] | None:
        """v scaled to length 1, or None for a zero vector (it has no direction)."""
        norm = float(np.linalg.norm(v))
        if norm == 0:
            return None
        return (v / norm).astype(np.float32)

    def lookup(
        self, query: str, *, top1: str, index_version: str, prompt_version: str, model: str
    ) -> Answer | None:
        v = self._unit(self.embedder.embed_query(query))
        if v is None:
            return None

        versions = (index_version, prompt_version, model)
        best_answer: Answer | None = None
        best_score = -1

        for entry in self._entries:
            if entry.versions != versions:
                continue
            if entry.top1 != top1:
                continue

            score = np.dot(entry.vector, v)
            if score > best_score:
                best_score, best_answer = score, entry.answer

        if best_answer is not None and best_score >= self.threshold:
            return best_answer
        return None

    def add(
        self,
        query: str,
        answer: Answer,
        *,
        top1: str,
        index_version: str,
        prompt_version: str,
        model: str,
        vector: NDArray[np.float32] | None = None,
    ) -> None:
        """Store an answer for later paraphrases; replaces the same question + versions."""
        # Reuse the vector the pipeline already computed for retrieval, if given.
        v = self._unit(vector if vector is not None else self.embedder.embed_query(query))
        if v is None:
            return  # a zero vector can never match anything; nothing worth storing

        question = normalize(query)
        versions = (index_version, prompt_version, model)

        # Replace, don't duplicate: drop an older entry for the same question and versions.
        # (The same question under different versions stays: it's a separate entry.)
        self._entries = [
            entry
            for entry in self._entries
            if not (entry.question == question and entry.versions == versions)
        ]
        self._entries.append(
            _Entry(question=question, versions=versions, top1=top1, vector=v, answer=answer)
        )

        # Over the limit: drop the oldest (front of the list).
        while len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def clear(self) -> None:
        """Forget every stored answer."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
