"""Contracts (typing.Protocol) that swappable parts must satisfy."""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from codesearch.schemas import Answer, ScoredChunk


@runtime_checkable
class Embedder(Protocol):
    """Interface for text embedding models."""

    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed documents into L2-normalised float32 vectors, shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        """Embed one search query, shape (dim,).

        Separate from ``embed`` because models like bge expect an instruction
        prefix on queries but not on documents.
        """
        ...


@runtime_checkable
class LLM(Protocol):
    """Interface for text generation providers."""

    async def generate(self, prompt: str) -> str:
        """Generate a complete text response for a given prompt."""
        ...

    def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream generated text tokens for a given prompt asynchronously."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Interface for document reranking models."""

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        """Rerank candidates against the query and return the best ``top_n``."""
        ...


@runtime_checkable
class Cache(Protocol):
    """Interface for exact/semantic answer caching mechanisms."""

    def get(self, key: str) -> Answer | None:
        """Retrieve a cached answer by key if present."""
        ...

    def set(self, key: str, answer: Answer) -> None:
        """Store an answer response in the cache under a given key."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Interface for search components retrieving relevant code chunks."""

    def retrieve(self, query: str) -> list[ScoredChunk]:
        """Retrieve relevant scored candidate chunks for a given query."""
        ...
