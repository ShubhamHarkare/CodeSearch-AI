"""Loading the models at startup instead of on the first user's request.

Each model class gets warm_up(): one tiny real inference, which loads the weights and
runs the first (slowest) forward pass. warm_up(pipeline) calls it on every component
that has one; fakes used in tests don't, and are skipped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from codesearch.explain.groundedness import NLIScorer
from codesearch.index.embedder import HashEmbedder, SentenceTransformerEmbedder
from codesearch.retrieval.rerank import CrossEncoderReranker, NoopReranker
from codesearch.warmup import warm_up

# =====================================================================
# Each model class: warm_up() runs one real (tiny) inference
# =====================================================================


class FakeSentenceModel:
    def __init__(self) -> None:
        self.encoded: list[list[str]] = []

    def encode(self, texts: list[str], **_: Any) -> Any:
        self.encoded.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return 4


def test_the_embedder_warm_up_encodes_once() -> None:
    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    model = FakeSentenceModel()
    embedder._model = model  # what _load() would have created
    embedder.warm_up()
    assert len(model.encoded) == 1  # a forward pass, not just loading the weights


def test_the_embedder_warm_up_loads_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    loads: list[int] = []

    def load() -> FakeSentenceModel:
        loads.append(1)
        embedder._model = FakeSentenceModel()
        return embedder._model

    monkeypatch.setattr(embedder, "_load", load)
    embedder.warm_up()
    assert loads  # the weights are in memory before the first request


def test_the_reranker_warm_up_scores_one_pair() -> None:
    seen: list[list[tuple[str, str]]] = []

    def scorer(pairs: list[tuple[str, str]]) -> list[float]:
        seen.append(list(pairs))
        return [0.0] * len(pairs)

    CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", scorer=scorer).warm_up()
    assert len(seen) == 1 and len(seen[0]) == 1


def test_the_nli_warm_up_runs_one_prediction() -> None:
    seen: list[list[tuple[str, str]]] = []

    def predict(pairs: list[tuple[str, str]]) -> Any:
        seen.append(list(pairs))
        return np.zeros((len(pairs), 3))

    NLIScorer(predict=predict).warm_up()
    assert len(seen) == 1 and len(seen[0]) == 1


# =====================================================================
# warm_up(pipeline): every component that can, once; timings for the log
# =====================================================================


class Recording:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def warm_up(self) -> None:
        self.calls += 1
        if self.fail:
            raise RuntimeError("could not download the model")


class Pipe:
    def __init__(self, embedder: Any = None, reranker: Any = None, grounder: Any = None) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.grounder = grounder


def test_every_model_is_warmed_once() -> None:
    e, r, g = Recording(), Recording(), Recording()
    seconds = warm_up(Pipe(e, r, g))
    assert (e.calls, r.calls, g.calls) == (1, 1, 1)
    assert set(seconds) == {"embedder", "reranker", "nli"}
    assert all(s >= 0 for s in seconds.values())


def test_missing_components_and_fakes_are_skipped() -> None:
    # No NLI (groundedness off), a fake embedder and the no-op reranker: nothing to load.
    assert warm_up(Pipe(HashEmbedder(), NoopReranker(), None)) == {}


def test_only_real_models_are_timed() -> None:
    r = Recording()
    assert set(warm_up(Pipe(HashEmbedder(), r, None))) == {"reranker"}
    assert r.calls == 1


def test_a_model_that_fails_to_load_stops_startup() -> None:
    # Better to fail at deploy time than to serve errors to the first users.
    with pytest.raises(RuntimeError, match="download"):
        warm_up(Pipe(Recording(), Recording(fail=True), Recording()))
