"""Builds concrete implementations from Settings. The only place that chooses.

Written so far (Phase 1a):
    - ``build_embedder(settings)``: "hash" -> HashEmbedder (tests, CI, quick
      runs); anything else -> SentenceTransformerEmbedder(model name).
    - ``load_index(settings)``: an IndexHandle holding the index on disk, or an
      empty handle if nothing has been built yet (the API then answers 503).

    - ``build_llm(settings)``: "ollama" -> OllamaLLM, "fake" -> FakeLLM.
    - ``build_reranker(settings)``: cross-encoder if a model is set, else no-op.

Still to come:
    - ``build_cache(settings, embedder) -> TieredCache``
    - ``build_pipeline(settings) -> AnswerPipeline`` wiring all of the above.

What it must do:
    - Load heavy models once and share them (embedder used by retrieval,
      semantic cache and drift monitoring).

Must not:
    - do request-time work. Factory builds; pipeline runs.

Phase: 1
"""

# from mlflow.tracking._model_registry.fluent import load_prompt
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


def build_pipeline(settings: Settings) -> AnswerPipeline:
    handle = load_index(settings)
    retriever = HybridRetriever(handle, build_embedder(settings), settings.retrieval)

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
    )
