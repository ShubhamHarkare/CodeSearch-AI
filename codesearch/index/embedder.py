"""Turns text into L2-normalised float32 vectors. Implements the Embedder protocol.

What goes here:
    - ``SentenceTransformerEmbedder``: the real model (default
      "BAAI/bge-small-en-v1.5", 384 dims). Loads lazily on first use, so
      importing this module or building the object costs nothing. ``embed_query``
      adds the bge query instruction; documents are embedded without it, which is
      how bge models are meant to be used.
    - ``HashEmbedder``: deterministic hashing-trick embedder for tests and CI.
      No downloads, no network.

Not yet: an Ollama embedder (optional, for local experiments).

Must not:
    - be imported anywhere except factory.py and tests (everything else uses
      the Embedder protocol).

Tested by: tests/unit/test_embedder.py (HashEmbedder) and
tests/unit/test_embedder_models.py (SentenceTransformerEmbedder, mocked + one slow real test).
Phase: 0 (HashEmbedder), 1 (real model)
"""

import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray


class HashEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self.dim: int = dim
        self.model_name: str = "hash"

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        embedding: NDArray[np.float32] = np.zeros((len(texts), self.dim), dtype=np.float32)

        for idx, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                c = (
                    int(hashlib.md5(word.encode(), usedforsecurity=False).hexdigest(), 16)
                    % self.dim
                )
                embedding[idx][c] += 1

        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        normalised: NDArray[np.float32] = embedding / norms
        return normalised

    def embed_query(self, text: str) -> NDArray[np.float32]:
        vector: NDArray[np.float32] = self.embed([text])[0]
        return vector


BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEmbedder:
    """A sentence-transformers model, loaded on first use."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        device: str | None = None,
        query_instruction: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        if query_instruction is None:  # bge models expect an instruction on queries only
            query_instruction = BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""
        self.query_instruction = query_instruction
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            # Imported here, not at the top: importing sentence_transformers loads
            # PyTorch, which would add seconds to every test run that imports this file.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        """Embedding size (loads the model if needed)."""
        return int(self._load().get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed documents: shape (len(texts), dim), each row L2-normalised."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result: NDArray[np.float32] = np.asarray(vectors, dtype=np.float32)
        return result

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed one search query (with the model's query instruction), shape (dim,)."""
        vector: NDArray[np.float32] = self.embed([self.query_instruction + text])[0]
        return vector
