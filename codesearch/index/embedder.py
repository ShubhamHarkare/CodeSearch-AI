"""Turns text into normalised vectors. Implements the Embedder protocol.

What goes here:
    - ``SentenceTransformerEmbedder(model_name, batch_size, device)``: default
      "BAAI/bge-small-en-v1.5"; adds the bge query instruction for queries via
      ``embed_query()``; returns float32, L2-normalised arrays.
    - ``OllamaEmbedder(model_name, base_url)``: optional, for local experiments.
    - ``HashEmbedder(dim=64)``: deterministic hashing-trick embedder for tests
      and CI. No downloads, no network.

What it must do:
    - Load the model lazily, once. Same text -> same vector.

Must not:
    - be imported anywhere except factory.py and tests (everything else uses
      the Embedder protocol).

Tested by: tests/unit/test_embedder.py (HashEmbedder: shape, norm 1,
determinism); a ``slow`` test for the real model.
Phase: 0 (HashEmbedder), 1 (real models)
"""

import hashlib

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
