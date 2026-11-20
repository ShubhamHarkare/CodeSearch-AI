"""Records exactly what an index was built from, so results are reproducible.

What goes here:
    - ``PIPELINE_VERSION``: bump by hand whenever mdx_clean / chunker logic changes,
      because then every chunk changes even if commit, model and settings did not.
    - ``Manifest``: corpus commit, embedding model + dim, chunk params, pipeline
      version, chunk count, build time, ``index_version`` and a changelog.
    - ``compute_index_version(...) -> str``: 12-hex-char hash of everything that
      affects the chunks. A new version means new cache keys, so stale cached
      answers can never be served after a re-index.
    - ``write_manifest(index_dir, manifest)``: write to a temp file, then
      ``os.replace`` it into place (atomic: readers never see a half-written file).
    - ``read_manifest(index_dir) -> Manifest | None``: None means "no index built
      yet"; a corrupt file raises instead of being silently ignored.

Tested by: tests/unit/test_manifest.py.
Phase: 1
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PIPELINE_VERSION = "1"
MANIFEST_NAME = "manifest.json"
BUILDS_DIR = "builds"  # index/builds/<build id>/: one folder per build
CURRENT_FILE = "CURRENT"  # index/CURRENT: the id of the live build


class ChunkParams(BaseModel):
    """Chunker settings that change what the chunks look like."""

    max_chars: int
    overlap: int
    strategy: Literal["v2", "v1"] = "v2"


class ChangeLogEntry(BaseModel):
    """What one build changed, compared with the index it replaced."""

    at: datetime
    corpus_sha: str
    added: int  # chunk ids new in this build
    removed: int  # chunk ids that disappeared
    unchanged: int


class Manifest(BaseModel):
    """Everything needed to say exactly what an index was built from."""

    index_version: str  # short hash of everything below that affects the chunks
    corpus_sha: str  # from CorpusSnapshot.commit_sha
    embedding_model: str  # "BAAI/bge-small-en-v1.5", or "hash" in tests
    embedding_dim: int  # 384 for bge-small
    chunk_params: ChunkParams
    pipeline_version: str  # PIPELINE_VERSION at build time
    chunk_count: int
    built_at: datetime
    changelog: list[ChangeLogEntry] = Field(default_factory=list)


def live_dir(index_dir: Path) -> Path:
    """The folder holding the live build: index/builds/<id> if CURRENT exists, else index/.

    Indexes built before build folders existed have their files directly in index/.
    """
    index_dir = Path(index_dir)
    pointer = index_dir / CURRENT_FILE
    if pointer.exists():
        build_id = pointer.read_text(encoding="utf-8").strip()
        return index_dir / BUILDS_DIR / build_id
    return index_dir


def set_live(index_dir: Path, build_id: str) -> None:
    """Point CURRENT at a finished build, atomically (temp file, then os.replace)."""
    index_dir = Path(index_dir)
    build_path = index_dir / BUILDS_DIR / build_id
    if not build_path.is_dir():
        raise FileNotFoundError(f"Build directory not found: {build_path}")

    tmp_path = index_dir / f"{CURRENT_FILE}.tmp"
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(build_id)
        f.flush()
        os.fsync(f.fileno())  # on disk before the rename, so a crash can't leave it empty
    os.replace(tmp_path, index_dir / CURRENT_FILE)


def compute_index_version(
    corpus_sha: str,
    embedding_model: str,
    chunk_params: ChunkParams,
    pipeline_version: str = PIPELINE_VERSION,
) -> str:
    """Short, stable hash of every input that affects the chunks or their vectors."""
    inputs: dict[str, object] = {
        "corpus_sha": corpus_sha,
        "embedding_model": embedding_model,
        "chunk_params": chunk_params.model_dump(),
        "pipeline_version": pipeline_version,
    }
    # sort_keys: the same inputs must always serialise to the same text.
    text = json.dumps(inputs, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def write_manifest(index_dir: Path, manifest: Manifest) -> None:
    """Write ``index_dir/manifest.json`` atomically (temp file, then os.replace)."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    tmp = index_dir / f"{MANIFEST_NAME}.tmp"
    tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    # Atomic swap: a reader sees the old file or the new one, never half of each.
    os.replace(tmp, index_dir / MANIFEST_NAME)


def read_manifest(index_dir: Path) -> Manifest | None:
    """Load the live build's manifest.json; None if no index has been built yet."""
    path = live_dir(index_dir) / MANIFEST_NAME  # the live build's manifest
    if not path.exists():
        return None
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
