from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class IndexSettings(BaseModel):
    """
    Settings related to database ingestion and index storage
    """

    react_dev_repo: str = "https://github.com/reactjs/react.dev.git"
    # A pinned commit keeps builds and evals reproducible; None means "latest main"
    # (what background re-indexing uses when admin.follow_upstream is on).
    react_dev_sha: str | None = "046f17d04295ba047bb5739026b4ac3110f17028"
    corpus_dir: Path = Path("data/")
    index_dir: Path = Path("index/")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_chars: int = 1000
    chunk_overlap: int = 0  # chunker doesn't apply overlap yet; see chunker.py docstring
    chunk_strategy: Literal["v2", "v1"] = "v2"


class RetrievalSettings(BaseModel):
    """Settings for vector search using BM25, and reranking parameters"""

    # dense won the Phase 1c ablation (eval/configs, results/): hybrid RRF and hybrid +
    # ms-marco rerank both scored lower MRR on the golden set.
    mode: Literal["dense", "bm25", "hybrid"] = "dense"
    bm25_k: int = 20
    dense_k: int = 20
    rrf_k: int = 60
    reranker_model: str | None = None
    rerank_top_n: int = 3
    abstain_threshold: float = 0.755
    history_turns: int = 3


class LLMSettings(BaseModel):
    """LLM provider and inference configurations."""

    provider: Literal["ollama", "fake"] = "fake"
    model: str = "gemma3:4b"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 120.0  # local models on a laptop CPU/GPU can take a while
    ollama_base_url: str = "http://localhost:11434"


class CacheSettings(BaseModel):
    """Redis and semantic caching configurations."""

    redis_url: str = "redis://localhost:6379/0"
    ttl_hours: int = 24
    semantic_threshold: float = 0.95
    semantic_max_entries: int = 1000
    enabled: bool = True
    semantic_enabled: bool = True


class MonitoringSettings(BaseModel):
    """Metrics tracking and drift detection thresholds."""

    window_size: int = 500
    drift_threshold: float = 0.2
    reference_questions: Path = Path("eval/golden.jsonl")
    drift_min_questions: int = 20


class AdminSettings(BaseModel):
    """Re-indexing: the webhook secret and the background schedule."""

    webhook_secret: SecretStr = SecretStr("")  # empty: the reindex webhook is disabled
    follow_upstream: bool = False  # background re-indexing fetches the latest main
    reindex_interval_hours: float = 6.0  # with follow_upstream; 0 turns the timer off


class ExplainSettings(BaseModel):
    trace_db: Path = Path("traces/trace.db")
    groundedness: bool = True
    supported_threshold: float = 0.7
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    partial_threshold: float = 0.4
    # Diagnostic: also score each sentence against EVERY source (finds mis-citations).
    # Off in production (costs sentences x chunks NLI pairs); the answer eval turns it on.
    check_all_sources: bool = False
    # Fitted by eval/fit_calibrator.py. Missing file -> answers have no confidence.
    calibrator_path: Path = Path("models/calibrator.json")


class Settings(BaseSettings):
    """Root configuration object for CodeSearch AI."""

    model_config = SettingsConfigDict(
        env_prefix="CODESEARCH_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    index: IndexSettings = Field(default_factory=IndexSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    admin: AdminSettings = Field(default_factory=AdminSettings)
    image_sha: str | None = None  # set via CODESEARCH_IMAGE_SHA (Docker build)
    explain: ExplainSettings = Field(default_factory=ExplainSettings)

    def with_overrides(self, **kwargs: Any) -> "Settings":
        """Return a new Settings instance with specified field overrides.

        Useful for unit tests, evaluation setups, or benchmarking runs.
        Supports passing sub-dictionaries (e.g., llm={"provider": "ollama"}).
        """
        current_data = self.model_dump()
        for key, value in kwargs.items():
            if (
                key in current_data
                and isinstance(current_data[key], dict)
                and isinstance(value, dict)
            ):
                current_data[key].update(value)
            else:
                current_data[key] = value
        return Settings.model_validate(current_data)


@lru_cache
def get_settings() -> Settings:
    """Return a cached, singleton instance of the application Settings."""
    return Settings()
