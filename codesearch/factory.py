"""Builds concrete implementations from Settings. The only place that chooses.

Written so far (Phase 1a):
    - ``build_embedder(settings)``: "hash" -> HashEmbedder (tests, CI, quick
      runs); anything else -> SentenceTransformerEmbedder(model name).
    - ``load_index(settings)``: an IndexHandle holding the index on disk, or an
      empty handle if nothing has been built yet (the API then answers 503).

Still to come:
    - ``build_llm(settings) -> LLM`` (groq | ollama | fake)
    - ``build_reranker(settings) -> Reranker`` (cross-encoder | noop)
    - ``build_cache(settings, embedder) -> TieredCache``
    - ``build_pipeline(settings) -> AnswerPipeline`` wiring all of the above.

What it must do:
    - Load heavy models once and share them (embedder used by retrieval,
      semantic cache and drift monitoring).

Must not:
    - do request-time work. Factory builds; pipeline runs.

Phase: 1
"""

from codesearch.config import Settings
from codesearch.index.embedder import HashEmbedder, SentenceTransformerEmbedder
from codesearch.index.handle import IndexHandle, IndexSnapshot
from codesearch.interfaces import Embedder

HASH_EMBEDDER = "hash"


def build_embedder(settings: Settings) -> Embedder:
    """The embedder named in settings.index.embedding_model."""
    if settings.index.embedding_model == HASH_EMBEDDER:
        return HashEmbedder()
    return SentenceTransformerEmbedder(model_name=settings.index.embedding_model)


def load_index(settings: Settings) -> IndexHandle:
    """A handle on the index in settings.index.index_dir (empty if not built yet)."""
    return IndexHandle(IndexSnapshot.open(settings.index.index_dir))
