"""Live re-indexing: build_index on a changed corpus while the old index stays in use.

Each build is written to its own folder (index/builds/<id>/) and becomes live only
when index/CURRENT is switched to it and the handle swaps. So a request holding the
old snapshot keeps a complete, unchanged index, and a failed build changes nothing.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from numpy.typing import NDArray

from codesearch.api.main import create_app
from codesearch.config import Settings
from codesearch.factory import load_index
from codesearch.index.embedder import HashEmbedder
from codesearch.index.handle import IndexHandle, IndexSnapshot
from codesearch.index.vector_store import VectorStore
from codesearch.ingest.build import build_index
from codesearch.ingest.manifest import BUILDS_DIR, CURRENT_FILE, live_dir, read_manifest
from tests.conftest import FakeReactDev

PAGE = "learn/state-a-components-memory.md"
PAGE_URL = "https://react.dev/learn/state-a-components-memory"
NEW_SENTENCE = "The quokkastate pattern keeps a zanzibar counter between renders."


class CountingEmbedder(HashEmbedder):
    """HashEmbedder that records every text it embeds (and can be told to fail)."""

    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.texts: list[str] = []
        self.fail = fail

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        if self.fail:
            raise RuntimeError("embedding service crashed")
        self.texts.extend(texts)
        return super().embed(texts)


def at(settings: Settings, repo: FakeReactDev) -> Settings:
    """Settings pointing at the fake repo's current head."""
    return settings.with_overrides(index={"react_dev_repo": repo.url, "react_dev_sha": repo.head})


def edit_page(repo: FakeReactDev) -> None:
    repo.commit({PAGE: repo.read(PAGE) + "\n\n" + NEW_SENTENCE + "\n"}, "edit one page")


def builds(index_dir: Path) -> list[str]:
    return sorted(p.name for p in (index_dir / BUILDS_DIR).iterdir() if p.is_dir())


# =====================================================================
# Layout: one folder per build, CURRENT points at the live one
# =====================================================================


def test_a_build_goes_into_its_own_folder_and_current_points_at_it(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    cfg = at(settings, fake_react_dev)
    report = build_index(cfg, HashEmbedder())
    index_dir = cfg.index.index_dir

    assert (index_dir / CURRENT_FILE).exists()
    assert live_dir(index_dir).parent == index_dir / BUILDS_DIR
    assert builds(index_dir) == [live_dir(index_dir).name]
    snapshot = IndexSnapshot.open(index_dir)  # callers still pass index_dir
    assert snapshot is not None
    assert snapshot.manifest.index_version == report.index_version


def test_load_index_opens_the_live_build(settings: Settings, fake_react_dev: FakeReactDev) -> None:
    cfg = at(settings, fake_react_dev)
    build_index(cfg, HashEmbedder())
    edit_page(fake_react_dev)
    report = build_index(at(settings, fake_react_dev), HashEmbedder())
    assert load_index(cfg).version == report.index_version


# =====================================================================
# Incremental: only the edited page is re-embedded
# =====================================================================


def test_editing_one_page_reembeds_only_that_pages_chunks(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    first = build_index(at(settings, fake_react_dev), HashEmbedder())
    edit_page(fake_react_dev)
    embedder = CountingEmbedder()
    second = build_index(at(settings, fake_react_dev), embedder)

    assert second.added >= 1
    assert second.removed >= 1  # the edited section's old chunk is gone
    assert second.unchanged == second.chunks - second.added
    assert second.unchanged > first.chunks // 2  # most of the corpus was reused
    assert len(embedder.texts) == second.added  # nothing else was embedded
    assert any(NEW_SENTENCE in text for text in embedder.texts)
    snapshot = IndexSnapshot.open(settings.index.index_dir)
    assert snapshot is not None
    changed = snapshot.vector_store.get(sorted(snapshot.vector_store.all_ids()))
    assert {c.url.split("#")[0] for c in changed if NEW_SENTENCE in c.text} == {PAGE_URL}


def test_a_rebuild_with_no_changes_embeds_nothing(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    cfg = at(settings, fake_react_dev)
    first = build_index(cfg, HashEmbedder())
    embedder = CountingEmbedder()
    again = build_index(cfg, embedder)
    assert (again.added, again.removed, again.unchanged) == (0, 0, first.chunks)
    assert embedder.texts == []
    assert again.index_version == first.index_version


# =====================================================================
# The swap: new requests see the new index, old snapshots stay whole
# =====================================================================


def test_the_new_sentence_is_searchable_after_the_swap(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    handle = IndexHandle()
    build_index(at(settings, fake_react_dev), HashEmbedder(), handle=handle)
    before = handle.snapshot()
    assert not before.bm25.search("quokkastate zanzibar", k=3) or all(
        score <= 0 for _, score in before.bm25.search("quokkastate zanzibar", k=3)
    )

    edit_page(fake_react_dev)
    build_index(at(settings, fake_react_dev), HashEmbedder(), handle=handle)
    after = handle.snapshot()
    assert after is not before
    top_id, top_score = after.bm25.search("quokkastate zanzibar", k=1)[0]
    assert top_score > 0
    assert NEW_SENTENCE in after.vector_store.get([top_id])[0].text


def test_a_snapshot_taken_before_the_rebuild_never_changes(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    """A request that started before the swap keeps a complete old index:
    its vector store and BM25 must still agree (the reviewer's question)."""
    handle = IndexHandle()
    build_index(at(settings, fake_react_dev), HashEmbedder(), handle=handle)
    old = handle.snapshot()
    old_ids = old.vector_store.all_ids()
    old_version = old.manifest.index_version

    edit_page(fake_react_dev)
    build_index(at(settings, fake_react_dev), HashEmbedder(), handle=handle)

    assert old.vector_store.all_ids() == old_ids  # the old build's files were not touched
    assert len(old.bm25) == len(old_ids)  # its vectors and BM25 still match
    assert old.manifest.index_version == old_version
    assert handle.snapshot().manifest.index_version != old_version
    hits = old.vector_store.query(HashEmbedder().embed_query("useState"), k=3)
    assert hits and all(chunk.id in old_ids for chunk, _ in hits)


# =====================================================================
# Failure and housekeeping
# =====================================================================


def test_a_failed_rebuild_leaves_the_live_index_alone(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    handle = IndexHandle()
    build_index(at(settings, fake_react_dev), HashEmbedder(), handle=handle)
    index_dir = settings.index.index_dir
    live_before = live_dir(index_dir)
    snapshot_before = handle.snapshot()

    edit_page(fake_react_dev)
    with pytest.raises(RuntimeError, match="crashed"):
        build_index(at(settings, fake_react_dev), CountingEmbedder(fail=True), handle=handle)

    assert live_dir(index_dir) == live_before
    assert handle.snapshot() is snapshot_before
    assert builds(index_dir) == [live_before.name]  # the half-built folder was removed


def test_old_builds_are_pruned_keeping_the_live_one_and_the_one_before(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    index_dir = settings.index.index_dir
    names = []
    for i in range(4):
        fake_react_dev.commit({PAGE: fake_react_dev.read(PAGE) + f"\n\nEdit number {i}.\n"})
        build_index(at(settings, fake_react_dev), HashEmbedder())
        names.append(live_dir(index_dir).name)
    # the previous build is kept for requests that still hold its snapshot
    assert builds(index_dir) == sorted(names[-2:])
    assert live_dir(index_dir).name == names[-1]


def test_an_index_built_in_the_old_layout_still_loads_and_upgrades(
    settings: Settings, fake_react_dev: FakeReactDev
) -> None:
    """Indexes built before build folders existed have their files directly in index/."""
    cfg = at(settings, fake_react_dev)
    build_index(cfg, HashEmbedder())
    index_dir = cfg.index.index_dir
    live = live_dir(index_dir)
    for part in live.iterdir():  # move the build's files up to index/ (the old layout)
        part.rename(index_dir / part.name)
    (index_dir / CURRENT_FILE).unlink()
    live.rmdir()

    old = IndexSnapshot.open(index_dir)
    assert old is not None and read_manifest(index_dir) is not None

    embedder = CountingEmbedder()
    report = build_index(cfg, embedder)  # upgrades: copies the old vectors, embeds nothing
    assert report.added == 0 and embedder.texts == []
    assert (index_dir / CURRENT_FILE).exists()
    reopened = IndexSnapshot.open(index_dir)
    assert reopened is not None
    assert VectorStore(live_dir(index_dir) / "chroma").count() == report.chunks


# =====================================================================
# The Phase 3 gate: edit a doc upstream, call the webhook, the answer changes
# =====================================================================

SECRET = "s3cret"


@asynccontextmanager
async def serve(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        yield client


def signed(body: bytes) -> dict[str, str]:
    digest = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    return {"X-Hub-Signature-256": f"sha256={digest}", "Content-Type": "application/json"}


async def test_a_doc_edit_reaches_answers_after_the_webhook(
    settings: Settings, fake_react_dev: FakeReactDev, tmp_path: Path
) -> None:
    cfg = at(settings, fake_react_dev).with_overrides(
        admin={"webhook_secret": SECRET, "follow_upstream": True, "reindex_interval_hours": 0},
        retrieval={"mode": "hybrid", "abstain_threshold": -2.0},  # always answer
        cache={"enabled": False},
        explain={"trace_db": tmp_path / "traces.db", "groundedness": False},
    )
    build_index(cfg, HashEmbedder())  # the index the server starts with (pinned commit)
    app = create_app(cfg)
    question = {"question": "What is the quokkastate zanzibar counter?"}

    async with serve(app) as c:
        before = (await c.get("/readyz")).json()["versions"]["index_version"]

        edit_page(fake_react_dev)  # a new commit on upstream main
        body = b'{"ref": "refs/heads/main"}'
        r = await c.post("/v1/admin/reindex", content=body, headers=signed(body))
        assert r.status_code == 202
        await app.state.reindexer.wait()

        status = (await c.get("/v1/admin/reindex")).json()
        assert status["last_error"] is None
        assert status["last_report"]["corpus_sha"] == fake_react_dev.head  # followed main
        after = (await c.get("/readyz")).json()["versions"]["index_version"]
        assert after != before  # new cache keys: no stale answers

        answer = (await c.post("/v1/ask", json=question)).json()
        trace = (await c.get(f"/v1/explain/{answer['trace_id']}")).json()
    texts = [candidate["chunk"]["text"] for candidate in trace["candidates"]]
    assert any(NEW_SENTENCE in text for text in texts)  # the edited page is retrieved now
