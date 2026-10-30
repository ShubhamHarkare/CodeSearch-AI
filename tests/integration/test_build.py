"""End-to-end tests for codesearch/ingest/build.py.

Each test builds a real index (git fetch -> clean -> chunk -> embed -> Chroma +
BM25 + manifest) from a fake react.dev repo holding the 22 mini-corpus pages.
No network: the repo is local and the embedder is the hash embedder.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from codesearch.config import Settings
from codesearch.errors import CorpusFetchError
from codesearch.index.embedder import HashEmbedder
from codesearch.index.handle import IndexHandle, IndexSnapshot
from codesearch.ingest.build import build_index
from codesearch.ingest.manifest import read_manifest
from tests.conftest import FakeReactDev


class CountingEmbedder(HashEmbedder):
    """HashEmbedder that remembers every text it was asked to embed."""

    def __init__(self, dim: int = 64, model_name: str = "hash") -> None:
        super().__init__(dim=dim)
        self.model_name = model_name
        self.embedded: list[str] = []

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        self.embedded.extend(texts)
        return super().embed(texts)


def at_commit(settings: Settings, repo: FakeReactDev, sha: str | None = None) -> Settings:
    return settings.with_overrides(
        index={"react_dev_repo": repo.url, "react_dev_sha": sha or repo.head}
    )


def source_paths(index_dir: Path) -> set[str]:
    snapshot = IndexSnapshot.open(index_dir)
    assert snapshot is not None
    chunks = snapshot.vector_store.get(sorted(snapshot.vector_store.all_ids()))
    return {chunk.source_path for chunk in chunks}


# =====================================================================
# First build
# =====================================================================


def test_first_build_creates_a_complete_index(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    embedder = CountingEmbedder()
    report = build_index(at_commit(settings, fake_react_dev), embedder)

    assert report.pages == 22
    assert report.chunks > 500
    assert report.added == report.chunks
    assert (report.removed, report.unchanged, report.full_rebuild) == (0, 0, False)
    assert len(embedder.embedded) == report.chunks

    snapshot = IndexSnapshot.open(settings.index.index_dir)
    assert snapshot is not None
    assert snapshot.vector_store.count() == len(snapshot.bm25) == report.chunks
    manifest = snapshot.manifest
    assert manifest.corpus_sha == fake_react_dev.head == report.corpus_sha
    assert manifest.embedding_model == "hash"
    assert manifest.embedding_dim == 64
    assert manifest.index_version == report.index_version
    assert len(manifest.changelog) == 1


def test_the_built_index_answers_searches(settings: Settings, fake_react_dev: FakeReactDev) -> None:
    embedder = CountingEmbedder()
    build_index(at_commit(settings, fake_react_dev), embedder)
    snapshot = IndexSnapshot.open(settings.index.index_dir)
    assert snapshot is not None

    top_id, _ = snapshot.bm25.search("createPortal", k=1)[0]
    (top,) = snapshot.vector_store.get([top_id])
    assert top.source_path == "reference/react-dom/createPortal.md"
    assert top.url == "https://react.dev/reference/react-dom/createPortal"

    hits = snapshot.vector_store.query(embedder.embed_query("updating state previous state"), k=3)
    assert hits and hits[0][1] > 0


# =====================================================================
# Incremental rebuilds
# =====================================================================


def test_second_run_with_no_changes_embeds_nothing(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    cfg = at_commit(settings, fake_react_dev)
    first = build_index(cfg, CountingEmbedder())
    embedder = CountingEmbedder()
    second = build_index(cfg, embedder)

    assert embedder.embedded == []
    assert (second.added, second.removed, second.unchanged) == (0, 0, first.chunks)
    assert second.index_version == first.index_version
    manifest = read_manifest(settings.index.index_dir)
    assert manifest is not None and len(manifest.changelog) == 2


def test_editing_one_page_re_embeds_only_that_page(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    first = build_index(at_commit(settings, fake_react_dev), CountingEmbedder())
    page = "reference/react/lazy.md"
    fake_react_dev.commit({page: fake_react_dev.read(page) + "\n\nA brand new closing note.\n"})

    embedder = CountingEmbedder()
    second = build_index(at_commit(settings, fake_react_dev), embedder)

    assert 1 <= second.added <= 3
    assert second.removed == second.added  # edited chunks swap one-for-one
    assert second.unchanged == first.chunks - second.added
    assert embedder.embedded and all(text.startswith("lazy") for text in embedder.embedded)
    assert second.index_version != first.index_version  # new commit -> new version


def test_a_page_deleted_upstream_disappears_from_the_index(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    build_index(at_commit(settings, fake_react_dev), CountingEmbedder())
    assert "reference/react-dom/createPortal.md" in source_paths(settings.index.index_dir)

    fake_react_dev.commit({"reference/react-dom/createPortal.md": None})
    report = build_index(at_commit(settings, fake_react_dev), CountingEmbedder())

    assert report.pages == 21
    assert report.removed > 0 and report.added == 0
    assert "reference/react-dom/createPortal.md" not in source_paths(settings.index.index_dir)


def test_a_new_page_upstream_is_added(settings: Settings, fake_react_dev: FakeReactDev) -> None:
    build_index(at_commit(settings, fake_react_dev), CountingEmbedder())
    fake_react_dev.commit(
        {"learn/brand-new.md": "---\ntitle: Brand New\n---\n\n## Hello {/*hello*/}\n\nHi.\n"}
    )
    embedder = CountingEmbedder()
    report = build_index(at_commit(settings, fake_react_dev), embedder)

    assert (report.pages, report.added, report.removed) == (23, 1, 0)
    assert embedder.embedded == ["Brand New > Hello\n\nHi."]


def test_building_an_older_commit_rolls_the_index_back(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    original = fake_react_dev.head
    first = build_index(at_commit(settings, fake_react_dev), CountingEmbedder())
    fake_react_dev.commit({"reference/react/lazy.md": None})
    build_index(at_commit(settings, fake_react_dev), CountingEmbedder())

    back = build_index(at_commit(settings, fake_react_dev, original), CountingEmbedder())

    assert back.chunks == first.chunks
    assert back.index_version == first.index_version
    assert "reference/react/lazy.md" in source_paths(settings.index.index_dir)


# =====================================================================
# Full rebuilds
# =====================================================================


def test_full_flag_re_embeds_everything(settings: Settings, fake_react_dev: FakeReactDev) -> None:
    cfg = at_commit(settings, fake_react_dev)
    first = build_index(cfg, CountingEmbedder())
    embedder = CountingEmbedder()
    report = build_index(cfg, embedder, full=True)

    assert report.full_rebuild
    assert report.added == first.chunks == len(embedder.embedded)


def test_a_different_embedding_model_forces_a_full_rebuild(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    """Chunk ids don't say which model embedded them, so a new model must re-embed all."""
    cfg = at_commit(settings, fake_react_dev)
    first = build_index(cfg, CountingEmbedder(dim=64, model_name="hash"))
    new_model = CountingEmbedder(dim=32, model_name="hash-32")
    report = build_index(cfg, new_model)

    assert report.full_rebuild
    assert report.added == first.chunks
    assert report.index_version != first.index_version
    snapshot = IndexSnapshot.open(settings.index.index_dir)
    assert snapshot is not None and snapshot.manifest.embedding_dim == 32
    assert snapshot.vector_store.query(new_model.embed_query("useState"), k=1)  # 32-d works


# =====================================================================
# Handle and failures
# =====================================================================


def test_a_given_handle_is_swapped_to_the_new_index(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    handle = IndexHandle()
    report = build_index(at_commit(settings, fake_react_dev), CountingEmbedder(), handle=handle)
    assert handle.ready
    assert handle.version == report.index_version


def test_a_failed_fetch_leaves_the_existing_index_untouched(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    cfg = at_commit(settings, fake_react_dev)
    first = build_index(cfg, CountingEmbedder())

    with pytest.raises(CorpusFetchError):
        build_index(at_commit(settings, fake_react_dev, "0" * 40), CountingEmbedder())

    manifest = read_manifest(settings.index.index_dir)
    assert manifest is not None and manifest.index_version == first.index_version
    assert IndexSnapshot.open(settings.index.index_dir) is not None
