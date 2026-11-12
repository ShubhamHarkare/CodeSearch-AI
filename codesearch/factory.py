"""Builds concrete implementations from Settings. The only place that chooses.

Written so far (Phase 1a):
    - ``build_embedder(settings)``: "hash" -> HashEmbedder (tests, CI, quick
      runs); anything else -> SentenceTransformerEmbedder(model name).
    - ``load_index(settings)``: an IndexHandle holding the index on disk, or an
      empty handle if nothing has been built yet (the API then answers 503).

    - ``build_llm(settings)``: "ollama" -> OllamaLLM, "fake" -> FakeLLM.
    - ``build_reranker(settings)``: cross-encoder if a model is set, else no-op.

    - ``build_cache(settings, embedder)``: TieredCache, or None if disabled.
    - ``build_pipeline(settings)``: AnswerPipeline wiring all of the above.

What it must do:
    - Load heavy models once and share them (embedder used by retrieval,
      semantic cache and drift monitoring).

Must not:
    - do request-time work. Factory builds; pipeline runs.

Phase: 1
"""

# from mlflow.tracking._model_registry.fluent import load_prompt
from codesearch.cache.exact import RedisExactCache
from codesearch.cache.memory import LRUCache
from codesearch.cache.semantic import SemanticCache
from codesearch.cache.tiered import TieredCache
from codesearch.config import Settings
from codesearch.generation.llm import FakeLLM, OllamaLLM
from codesearch.generation.prompts import load_prompt
from codesearch.index.embedder import HashEmbedder, SentenceTransformerEmbedder
from codesearch.index.handle import IndexHandle, IndexSnapshot
from codesearch.interfaces import LLM, Embedder, Reranker
from codesearch.pipeline import AnswerPipeline
from codesearch.retrieval.hybrid import HybridRetriever
from codesearch.retrieval.rerank import CrossEncoderReranker, NoopReranker

HASH_EMBEDDER = "hash"


def build_embedder(settings: Settings) -> Embedder:
    """The embedder named in settings.index.embedding_model."""
    if settings.index.embedding_model == HASH_EMBEDDER:
        return HashEmbedder()
    return SentenceTransformerEmbedder(model_name=settings.index.embedding_model)


def load_index(settings: Settings) -> IndexHandle:
    """A handle on the index in settings.index.index_dir (empty if not built yet)."""
    return IndexHandle(IndexSnapshot.open(settings.index.index_dir))


def build_reranker(settings: Settings) -> Reranker:
    """A cross-encoder if settings.retrieval.reranker_model is set, else a no-op."""
    model = settings.retrieval.reranker_model
    if model:
        return CrossEncoderReranker(model)
    return NoopReranker()


def build_llm(settings: Settings) -> LLM:
    """The LLM named in settings.llm.provider: a local Ollama model, or the fake."""
    llm = settings.llm
    if llm.provider == "ollama":
        return OllamaLLM(
            llm.model,
            llm.ollama_base_url,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            timeout_s=llm.timeout_s,
        )
    return FakeLLM()


def build_cache(settings: Settings, embedder: Embedder) -> TieredCache | None:
    """Redis (with an in-memory fallback) plus the semantic tier, or None if disabled."""
    c = settings.cache
    if not c.enabled:
        return None
    semantic = (
        SemanticCache(embedder, c.semantic_threshold, c.semantic_max_entries)
        if c.semantic_enabled
        else None
    )
    return TieredCache(
        exact=RedisExactCache(c.redis_url, c.ttl_hours),  # takes hours, not seconds
        fallback=LRUCache(ttl_seconds=c.ttl_hours * 3600),  # first positional is max_items
        semantic=semantic,
    )


def build_pipeline(settings: Settings) -> AnswerPipeline:
    handle = load_index(settings)
    embedder = build_embedder(settings)  # shared by retrieval and the semantic cache
    retriever = HybridRetriever(handle, embedder, settings.retrieval)

    reranker = build_reranker(settings)
    llm = build_llm(settings)

    answer_prompt = load_prompt("answer")
    condense_prompt = load_prompt("condense")

    return AnswerPipeline(
        handle=handle,
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        answer_prompt=answer_prompt,
        condense_prompt=condense_prompt,
        settings=settings,
        cache=build_cache(settings, embedder),
    )
