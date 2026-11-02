"""build_index with chunk_strategy="v1": the baseline index (fake repo, hash embedder)."""

from __future__ import annotations

import math

from codesearch.config import Settings
from codesearch.index.embedder import HashEmbedder
from codesearch.index.handle import IndexSnapshot
from codesearch.ingest.build import build_index
from codesearch.ingest.manifest import read_manifest
from codesearch.ingest.mdx_clean import clean_mdx
from tests.conftest import FakeReactDev


def configure(settings: Settings, repo: FakeReactDev, strategy: str, sub: str) -> Settings:
    return settings.with_overrides(
        index={
            "react_dev_repo": repo.url,
            "react_dev_sha": repo.head,
            "chunk_strategy": strategy,
            "index_dir": settings.index.index_dir / sub,
        }
    )


def stored_chunks(settings: Settings) -> list:  # type: ignore[type-arg]
    snap = IndexSnapshot.open(settings.index.index_dir)
    assert snap is not None
    return snap.vector_store.get(sorted(snap.vector_store.all_ids()))


def test_v1_build_stores_only_anchorless_fixed_chunks(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    cfg = configure(settings, fake_react_dev, "v1", "v1")
    report = build_index(cfg, HashEmbedder())

    expected = 0
    for path in fake_react_dev.content.rglob("*.md"):
        if path.name in {"README.md", "LICENSE-DOCS.md"}:
            continue
        body = clean_mdx(path.read_text(encoding="utf-8")).body
        expected += math.ceil(len(body) / 1000) if body.strip() else 0
    # whitespace-only pieces are dropped, so allow the count to be slightly lower
    assert expected - 5 <= report.chunks <= expected

    chunks = stored_chunks(cfg)
    assert len(chunks) == report.chunks
    assert all(c.anchor == "" and c.header_path == () for c in chunks)
    assert all(len(c.text) <= 1000 for c in chunks)


def test_v1_manifest_records_the_strategy(settings: Settings, fake_react_dev: FakeReactDev) -> None:
    cfg = configure(settings, fake_react_dev, "v1", "v1")
    build_index(cfg, HashEmbedder())
    manifest = read_manifest(cfg.index.index_dir)
    assert manifest is not None
    assert manifest.chunk_params.strategy == "v1"


def test_v1_and_v2_indexes_side_by_side(settings: Settings, fake_react_dev: FakeReactDev) -> None:
    v1 = configure(settings, fake_react_dev, "v1", "v1")
    v2 = configure(settings, fake_react_dev, "v2", "v2")
    r1 = build_index(v1, HashEmbedder())
    r2 = build_index(v2, HashEmbedder())

    assert r1.index_version != r2.index_version
    assert any(c.anchor for c in stored_chunks(v2))
    assert not any(c.anchor for c in stored_chunks(v1))


def test_switching_strategy_in_one_dir_replaces_every_chunk(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    v2 = configure(settings, fake_react_dev, "v2", "shared")
    v1 = configure(settings, fake_react_dev, "v1", "shared")
    first = build_index(v2, HashEmbedder())
    second = build_index(v1, HashEmbedder())

    assert second.unchanged == 0
    assert second.removed == first.chunks
    assert not any(c.anchor for c in stored_chunks(v1))
