"""Tests for SentenceTransformerEmbedder.

The model is mocked, so these run offline in milliseconds. One test marked
``slow`` downloads the real bge-small model; CI skips it (-m "not slow").
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from codesearch.index.embedder import BGE_QUERY_INSTRUCTION, SentenceTransformerEmbedder
from codesearch.interfaces import Embedder

pytest.importorskip("sentence_transformers")


def fake_model(dim: int = 384) -> MagicMock:
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim
    model.encode.side_effect = lambda texts, **_: np.ones((len(texts), dim), dtype=np.float64)
    return model


def test_model_is_not_loaded_until_first_use() -> None:
    with patch("sentence_transformers.SentenceTransformer") as st:
        st.return_value = fake_model()
        embedder = SentenceTransformerEmbedder()
        st.assert_not_called()

        embedder.embed(["a", "b"])
        embedder.embed(["c"])
        st.assert_called_once_with("BAAI/bge-small-en-v1.5", device=None)  # loaded once


def test_documents_are_embedded_without_the_query_instruction() -> None:
    with patch("sentence_transformers.SentenceTransformer") as st:
        st.return_value = model = fake_model()
        SentenceTransformerEmbedder().embed(["doc text"])
        assert model.encode.call_args.args[0] == ["doc text"]
        assert model.encode.call_args.kwargs["normalize_embeddings"] is True


def test_queries_get_the_bge_instruction() -> None:
    with patch("sentence_transformers.SentenceTransformer") as st:
        st.return_value = model = fake_model()
        vector = SentenceTransformerEmbedder().embed_query("find relevant docs")
        assert model.encode.call_args.args[0] == [BGE_QUERY_INSTRUCTION + "find relevant docs"]
        assert vector.shape == (384,)


def test_non_bge_models_get_no_instruction() -> None:
    with patch("sentence_transformers.SentenceTransformer") as st:
        st.return_value = model = fake_model()
        SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2").embed_query("q")
        assert model.encode.call_args.args[0] == ["q"]


def test_output_is_float32_with_the_right_shape() -> None:
    with patch("sentence_transformers.SentenceTransformer") as st:
        st.return_value = fake_model(dim=8)
        out = SentenceTransformerEmbedder().embed(["a", "b", "c"])
        assert out.dtype == np.float32
        assert out.shape == (3, 8)


def test_embedding_nothing_returns_an_empty_matrix() -> None:
    with patch("sentence_transformers.SentenceTransformer") as st:
        st.return_value = model = fake_model(dim=8)
        out = SentenceTransformerEmbedder().embed([])
        assert out.shape == (0, 8)
        model.encode.assert_not_called()


def test_satisfies_the_embedder_protocol() -> None:
    with patch("sentence_transformers.SentenceTransformer") as st:
        st.return_value = fake_model()
        assert isinstance(SentenceTransformerEmbedder(), Embedder)


@pytest.mark.slow
def test_real_bge_small_model() -> None:
    """Downloads ~130 MB the first time. Run with: uv run pytest -m slow"""
    embedder = SentenceTransformerEmbedder()
    docs = embedder.embed(["useState adds state to a component", "CSS grid layout"])
    query = embedder.embed_query("how do I add state to a React component?")
    assert docs.shape == (2, 384)
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-4)
    assert float(docs[0] @ query) > float(docs[1] @ query)
