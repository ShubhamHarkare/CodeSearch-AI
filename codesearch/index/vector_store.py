"""Small wrapper around Chroma. The only module that imports chromadb.

What goes here:
    - ``VectorStore(path, collection_name)``: a persistent Chroma collection using
      COSINE distance (Chroma's default is L2, which is wrong for our embeddings).
    - ``upsert(chunks, vectors)``: insert or update by chunk id, in batches of at
      most ``client.get_max_batch_size()`` (5,461 in Chroma 1.5).
    - ``reset()``: drop everything, including the vector size (new embedding model).
    - ``delete(ids)``, ``all_ids() -> set[str]``, ``count() -> int``: what
      ``ingest/build.py`` needs to re-embed only new chunks and drop removed ones.
    - ``get(ids) -> list[Chunk]``: in the order asked for (Chroma doesn't promise it).
    - ``query(vector, k) -> list[tuple[Chunk, float]]``: best first, scored as
      cosine SIMILARITY (1 - Chroma's distance), so higher always means better.

Chroma quirks handled here (checked against chromadb 1.5.9):
    - collection names must be 3-512 characters;
    - an empty list is rejected as a metadata value, so ``header_path`` is
      stored as a JSON string;
    - duplicate ids in one call raise, so ``get`` de-duplicates and ``upsert``
      reports them with a clear message.

Tested by: tests/unit/test_vector_store.py (real Chroma in tmp_path).
Phase: 1
"""

import json
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings
from numpy.typing import NDArray

from codesearch.log import get_logger
from codesearch.schemas import Chunk

log = get_logger(__name__)


class VectorStore:
    """Persistent dense-vector store for Chunks (Chroma, cosine distance)."""

    def __init__(self, path: Path, collection_name: str = "react_docs") -> None:
        self.path = Path(path)
        self.client = chromadb.PersistentClient(
            path=str(self.path), settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name
        self.collection = self._open_collection()

    def _open_collection(self) -> Any:
        return self.client.get_or_create_collection(
            name=self.collection_name, configuration={"hnsw": {"space": "cosine"}}
        )

    def reset(self) -> None:
        """Delete every chunk AND the collection's vector size (for a new embedding model)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._open_collection()

    def upsert(self, chunks: list[Chunk], vectors: NDArray[np.float32]) -> None:
        """Insert or update chunks with their vectors (one row of ``vectors`` per chunk)."""
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        ids = [chunk.id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk ids in one upsert")
        if not chunks:
            return

        batch_size = self.client.get_max_batch_size()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                embeddings=vectors[start : start + batch_size].tolist(),
                documents=[chunk.text for chunk in batch],
                metadatas=[_to_metadata(chunk) for chunk in batch],
            )
        log.info("vector_store.upserted", chunks=len(chunks), total=self.count())

    def delete(self, ids: list[str]) -> None:
        """Delete chunks by id; empty lists and unknown ids are fine."""
        if ids:
            self.collection.delete(ids=list(ids))

    def all_ids(self) -> set[str]:
        """Every stored chunk id (fetches ids only, no vectors or text)."""
        return set(self.collection.get(include=[])["ids"])

    def count(self) -> int:
        """Number of stored chunks."""
        return int(self.collection.count())

    def get(self, ids: list[str]) -> list[Chunk]:
        """Fetch chunks by id, in the order asked for; unknown ids are skipped."""
        unique_ids = list(dict.fromkeys(ids))  # Chroma raises on duplicate ids
        if not unique_ids:
            return []
        result = self.collection.get(ids=unique_ids, include=["documents", "metadatas"])
        by_id = {
            chunk_id: _from_record(chunk_id, text, metadata)
            for chunk_id, text, metadata in zip(
                result["ids"], result["documents"] or [], result["metadatas"] or [], strict=True
            )
        }
        return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]

    def query(self, vector: NDArray[np.float32], k: int) -> list[tuple[Chunk, float]]:
        """The ``k`` most similar chunks, best first, with cosine similarity scores."""
        if k <= 0 or self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        return [
            (_from_record(chunk_id, text, metadata), 1.0 - float(distance))
            for chunk_id, text, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]


def _to_metadata(chunk: Chunk) -> dict[str, str | int]:
    """Chunk fields Chroma doesn't already hold as id / document."""
    return {
        "url": chunk.url,
        "anchor": chunk.anchor,
        "title": chunk.title,
        "header_path": json.dumps(list(chunk.header_path)),
        "source_path": chunk.source_path,
        "start_line": chunk.start_line,
    }


def _from_record(chunk_id: str, text: str | None, metadata: Any) -> Chunk:
    """Rebuild a Chunk from one Chroma record."""
    return Chunk(
        id=chunk_id,
        url=str(metadata["url"]),
        anchor=str(metadata["anchor"]),
        title=str(metadata["title"]),
        header_path=tuple(json.loads(str(metadata["header_path"]))),
        text=text or "",
        source_path=str(metadata["source_path"]),
        start_line=int(metadata["start_line"]),
    )
