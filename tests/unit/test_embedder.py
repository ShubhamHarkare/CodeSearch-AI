"""Tests for HashEmbedder in codesearch/index/embedder.py."""

import numpy as np

from codesearch.index.embedder import HashEmbedder
from codesearch.interfaces import Embedder

# =====================================================================
# HashEmbedder Tests (Phase 0 - Quick CI Tests)
# =====================================================================


def test_hash_embedder_shape_and_dtype():
    embedder = HashEmbedder(dim=64)
    texts = ["hello world", "foo bar baz"]

    result = embedder.embed(texts)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 64)
    assert result.dtype == np.float32


def test_hash_embedder_l2_norm():
    embedder = HashEmbedder(dim=64)
    texts = ["machine learning and vector search", "python pytest"]

    vectors = embedder.embed(texts)

    # Check that the L2 norm of each row vector is ~1.0
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_hash_embedder_determinism():
    embedder = HashEmbedder(dim=64)
    text = "the quick brown fox jumps over the lazy dog"

    vec1 = embedder.embed([text])
    vec2 = embedder.embed([text])

    np.testing.assert_array_equal(vec1, vec2)


def test_hash_embedder_embed_query_consistency():
    embedder = HashEmbedder(dim=64)
    text = "search query string"

    batch_vec = embedder.embed([text])[0]
    query_vec = embedder.embed_query(text)

    assert query_vec.shape == (64,)
    np.testing.assert_array_equal(batch_vec, query_vec)


def test_hash_embedder_case_insensitivity():
    embedder = HashEmbedder(dim=64)

    vec1 = embedder.embed(["HELLO WORLD"])
    vec2 = embedder.embed(["hello world"])

    np.testing.assert_array_equal(vec1, vec2)


# --- Edge Cases for HashEmbedder ---


def test_hash_embedder_empty_list():
    embedder = HashEmbedder(dim=64)
    result = embedder.embed([])

    assert result.shape == (0, 64)
    assert result.dtype == np.float32


def test_hash_embedder_empty_string_and_whitespace():
    embedder = HashEmbedder(dim=64)
    texts = ["", "   ", "\n\t"]

    vectors = embedder.embed(texts)

    assert vectors.shape == (3, 64)
    # Zero text shouldn't cause division by zero or NaN values
    assert not np.isnan(vectors).any()
    # Should result in zero vectors
    np.testing.assert_array_equal(vectors, np.zeros((3, 64), dtype=np.float32))


def test_hash_embedder_custom_dimension():
    for dim in [1, 16, 128, 1536]:
        embedder = HashEmbedder(dim=dim)
        res = embedder.embed(["test text"])
        assert res.shape == (1, dim)


def test_hash_embedder_unicode_and_punctuation():
    embedder = HashEmbedder(dim=64)
    texts = ["🚀 Python 3.12!", "日本語のテキスト", "åäö Éléphant"]

    vectors = embedder.embed(texts)
    assert vectors.shape == (3, 64)
    assert not np.isnan(vectors).any()


def test_hash_embedder_satisfies_embedder_protocol():
    assert isinstance(HashEmbedder(), Embedder)


def test_hash_embedder_shared_words_are_more_similar():
    embedder = HashEmbedder(dim=64)
    base, related, unrelated = embedder.embed(["usestate hook", "usestate state", "routing server"])
    assert float(base @ related) > float(base @ unrelated)
