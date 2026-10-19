"""Phase 1 tests for the real embedders (SentenceTransformerEmbedder, OllamaEmbedder).

Skipped until those classes exist in codesearch/index/embedder.py. When you write
them, delete the skip line below. Note: the Ollama test patches ``requests.post``;
either add ``requests`` as a dependency or patch the HTTP client you actually use.
"""

import pytest

pytest.skip("Phase 1: real embedders not written yet", allow_module_level=True)

from unittest.mock import MagicMock, patch  # noqa: E402

import numpy as np  # noqa: E402

from codesearch.index.embedder import OllamaEmbedder, SentenceTransformerEmbedder  # noqa: E402

# =====================================================================
# Real Model Tests (Phase 1)
# =====================================================================


def test_sentence_transformer_lazy_loading():
    """Verify model is not loaded during initialization."""
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-small-en-v1.5")
        mock_st.assert_not_called()

        # Accessing embed triggers load
        embedder.embed(["test"])
        mock_st.assert_called_once()


def test_sentence_transformer_bge_query_instruction():
    """Verify BGE query prefix instruction is prepended to queries."""
    with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-small-en-v1.5")
        embedder.embed_query("find relevant docs")

        # Verify query instruction was prepended in encode call
        args, _ = mock_model.encode.call_args
        encoded_text = args[0][0]
        assert encoded_text.startswith("Represent this sentence for searching relevant passages:")
        assert "find relevant docs" in encoded_text


@pytest.mark.slow
def test_sentence_transformer_integration():
    """Real model download and execution test."""
    embedder = SentenceTransformerEmbedder()
    vectors = embedder.embed(["This is a test document."])

    assert vectors.dtype == np.float32
    assert vectors.shape[0] == 1
    # Check L2 normalization
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-4)


def test_ollama_embedder_mocked():
    """Verify Ollama integration formatting and output normalization."""
    mock_response = {
        "embeddings": [[3.0, 4.0, 0.0]]  # L2 norm = 5.0 -> Normalized = [0.6, 0.8, 0.0]
    }

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response

        embedder = OllamaEmbedder(model_name="nomic-embed-text", base_url="http://localhost:11434")
        vectors = embedder.embed(["test text"])

        assert vectors.shape == (1, 3)
        np.testing.assert_allclose(vectors, [[0.6, 0.8, 0.0]], rtol=1e-5)
