"""Shared pytest fixtures. Tests must run offline in seconds.

Fixtures to write:
    - ``settings(tmp_path)``: Settings with data/ and index/ under tmp_path,
      provider="fake", Redis disabled.
    - ``hash_embedder``: HashEmbedder(dim=64).
    - ``fake_llm``: FakeLLM with default citing answers.
    - ``mini_corpus_path``: tests/fixtures/mini_corpus.
    - ``mini_index`` (session scope): build the index once from the mini corpus.
    - ``pipeline``: AnswerPipeline built by factory.build_pipeline(settings).

Rules: no network, no real models, no Redis (use fakeredis), unless a test is
marked ``slow`` or ``needs_llm``.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real env vars out of every test."""
    for key in list(os.environ):
        if key.startswith("CODESEARCH_"):
            monkeypatch.delenv(key)
