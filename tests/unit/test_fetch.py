"""Tests for codesearch/ingest/fetch.py.

No network: every test builds a tiny fake "react.dev" git repo in tmp_path and
fetches from it with a file:// URL (plain local paths would ignore --depth).
"""

import subprocess
from datetime import UTC
from pathlib import Path

import pytest

from codesearch.errors import CodeSearchError, CorpusFetchError
from codesearch.ingest import fetch
from codesearch.ingest.fetch import fetch_corpus, list_pages, run_git

# =====================================================================
# Helpers: a fake upstream repo
# =====================================================================


def git(repo: Path, *args: str) -> str:
    """Run git in ``repo`` with a throwaway identity (CI runners have none)."""
    result = subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def commit(repo: Path, files: dict[str, str | None], message: str) -> str:
    """Write (or delete, when the value is None) files, commit, return the new SHA."""
    for rel, content in files.items():
        path = repo / rel
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "--quiet", "-m", message)
    return git(repo, "rev-parse", "HEAD")


FIRST_FILES: dict[str, str | None] = {
    "README.md": "repo readme, outside src/content",
    "src/content/learn/index.md": "---\ntitle: Learn\n---\nversion one\n",
    "src/content/reference/react/useState.md": "---\ntitle: useState\n---\n",
    "src/content/blog/2024/post.mdx": "---\ntitle: Post\n---\n",
    "src/content/README.md": "not a page",
    "src/content/LICENSE-DOCS.md": "license, not a page",
    "src/content/images/diagram.png": "binary-ish",
    "src/content/notes.txt": "not markdown",
}
EXPECTED_PAGES = ["blog/2024/post.mdx", "learn/index.md", "reference/react/useState.md"]


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init", "--quiet", "-b", "main")
    # Lets tests fetch an older commit by SHA (GitHub allows this by default).
    git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")
    commit(repo, FIRST_FILES, "first")
    return repo


def url(repo: Path) -> str:
    return f"file://{repo}"


def rel_pages(snapshot: fetch.CorpusSnapshot) -> list[str]:
    return [p.relative_to(snapshot.content_root).as_posix() for p in snapshot.page_paths]


# =====================================================================
# fetch_corpus: happy paths
# =====================================================================


def test_first_fetch_checks_out_the_requested_commit(upstream: Path, tmp_path: Path) -> None:
    sha = git(upstream, "rev-parse", "HEAD")
    dest = tmp_path / "data" / "react.dev"  # parent folders don't exist yet

    snapshot = fetch_corpus(url(upstream), sha, dest)

    assert snapshot.commit_sha == sha
    assert snapshot.repo_dir == dest
    assert snapshot.content_root == dest / "src" / "content"
    assert rel_pages(snapshot) == EXPECTED_PAGES
    assert snapshot.fetched_at.tzinfo is UTC


def test_pinned_older_commit_gives_old_content(upstream: Path, tmp_path: Path) -> None:
    old_sha = git(upstream, "rev-parse", "HEAD")
    commit(upstream, {"src/content/learn/index.md": "---\ntitle: Learn\n---\nversion two\n"}, "2")

    snapshot = fetch_corpus(url(upstream), old_sha, tmp_path / "dest")

    assert snapshot.commit_sha == old_sha
    assert "version one" in (snapshot.content_root / "learn/index.md").read_text()


def test_sha_none_fetches_latest_main(upstream: Path, tmp_path: Path) -> None:
    latest = commit(upstream, {"src/content/learn/new-page.md": "---\ntitle: New\n---\n"}, "2")

    snapshot = fetch_corpus(url(upstream), None, tmp_path / "dest")

    assert snapshot.commit_sha == latest
    assert "learn/new-page.md" in rel_pages(snapshot)


def test_refetch_moves_forward_and_drops_deleted_pages(upstream: Path, tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    first = fetch_corpus(url(upstream), git(upstream, "rev-parse", "HEAD"), dest)
    assert "reference/react/useState.md" in rel_pages(first)

    second_sha = commit(
        upstream,
        {
            "src/content/reference/react/useState.md": None,  # page deleted upstream
            "src/content/learn/index.md": "---\ntitle: Learn\n---\nversion two\n",
        },
        "2",
    )
    second = fetch_corpus(url(upstream), second_sha, dest)  # same dest: the update path

    assert second.commit_sha == second_sha
    assert "reference/react/useState.md" not in rel_pages(second)
    assert not (dest / "src/content/reference/react/useState.md").exists()
    assert "version two" in (dest / "src/content/learn/index.md").read_text()


def test_refetch_of_the_same_commit_is_harmless(upstream: Path, tmp_path: Path) -> None:
    sha = git(upstream, "rev-parse", "HEAD")
    dest = tmp_path / "dest"
    first = fetch_corpus(url(upstream), sha, dest)
    second = fetch_corpus(url(upstream), sha, dest)
    assert first.commit_sha == second.commit_sha
    assert first.page_paths == second.page_paths


def test_local_edits_in_the_checkout_are_overwritten(upstream: Path, tmp_path: Path) -> None:
    """The checkout is a cache of upstream: hand edits must never leak into the index."""
    sha = git(upstream, "rev-parse", "HEAD")
    dest = tmp_path / "dest"
    fetch_corpus(url(upstream), sha, dest)
    (dest / "src/content/learn/index.md").write_text("edited by hand")

    fetch_corpus(url(upstream), sha, dest)

    assert "version one" in (dest / "src/content/learn/index.md").read_text()


def test_changed_repo_url_is_picked_up(upstream: Path, tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    fetch_corpus(url(upstream), None, dest)

    mirror = tmp_path / "mirror"
    git(tmp_path, "clone", "--quiet", str(upstream), str(mirror))
    mirror_sha = commit(mirror, {"src/content/learn/mirror.md": "---\ntitle: M\n---\n"}, "m")

    snapshot = fetch_corpus(url(mirror), None, dest)
    assert snapshot.commit_sha == mirror_sha


# =====================================================================
# fetch_corpus: failures become CorpusFetchError
# =====================================================================


def test_unknown_sha_raises(upstream: Path, tmp_path: Path) -> None:
    with pytest.raises(CorpusFetchError, match="fetch"):
        fetch_corpus(url(upstream), "0" * 40, tmp_path / "dest")


@pytest.mark.parametrize("bad_sha", ["main", "abc123", "A" * 40, "0" * 39, ""])
def test_malformed_sha_is_rejected_before_running_git(tmp_path: Path, bad_sha: str) -> None:
    dest = tmp_path / "dest"
    with pytest.raises(CorpusFetchError, match="40-character"):
        fetch_corpus("file:///does/not/matter", bad_sha, dest)
    assert not dest.exists()  # nothing was created


def test_unreachable_repo_raises(tmp_path: Path) -> None:
    with pytest.raises(CorpusFetchError):
        fetch_corpus(url(tmp_path / "no-such-repo"), None, tmp_path / "dest")


def test_non_empty_non_git_dest_is_refused(upstream: Path, tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "important.txt").write_text("someone's files")

    with pytest.raises(CorpusFetchError, match="not a git repository"):
        fetch_corpus(url(upstream), None, dest)
    assert (dest / "important.txt").read_text() == "someone's files"


def test_commit_without_src_content_raises(tmp_path: Path) -> None:
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init", "--quiet", "-b", "main")
    commit(repo, {"README.md": "no docs here"}, "first")

    with pytest.raises(CorpusFetchError, match="no src/content"):
        fetch_corpus(url(repo), None, tmp_path / "dest")


def test_commit_without_pages_raises(tmp_path: Path) -> None:
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init", "--quiet", "-b", "main")
    commit(repo, {"src/content/README.md": "only a readme"}, "first")

    with pytest.raises(CorpusFetchError, match=r"no \.md/\.mdx pages"):
        fetch_corpus(url(repo), None, tmp_path / "dest")


def test_fetch_error_is_the_shared_codesearch_error() -> None:
    error = CorpusFetchError("boom")
    assert isinstance(error, CodeSearchError)  # so the API can map it to an HTTP status
    assert "boom" not in error.public_message


# =====================================================================
# list_pages
# =====================================================================


def test_list_pages_filters_and_sorts(tmp_path: Path) -> None:
    root = tmp_path / "content"
    for rel in ["b/z.md", "a/y.mdx", "a/x.md", "README.md", "LICENSE-DOCS.md", "c.txt", "d.MD"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x")
    (root / "folder.md").mkdir()  # a directory named like a page is not a page

    pages = [p.relative_to(root).as_posix() for p in list_pages(root)]
    assert pages == ["a/x.md", "a/y.mdx", "b/z.md"]


def test_list_pages_on_missing_dir_is_empty(tmp_path: Path) -> None:
    assert list_pages(tmp_path / "nope") == []


def test_list_pages_on_the_real_mini_corpus() -> None:
    root = Path(__file__).parents[1] / "fixtures" / "mini_corpus"
    pages = list_pages(root)
    assert len(pages) == 22
    assert all(p.name not in {"README.md", "LICENSE-DOCS.md"} for p in pages)


# =====================================================================
# run_git
# =====================================================================


def test_run_git_returns_stripped_stdout(upstream: Path) -> None:
    assert run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=upstream) == "main"


def test_run_git_failure_includes_git_message(tmp_path: Path) -> None:
    with pytest.raises(CorpusFetchError, match="not a git repository"):
        run_git(["status"], cwd=tmp_path)


def test_run_git_without_git_installed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))  # no git binary anywhere on PATH
    with pytest.raises(CorpusFetchError, match="not installed"):
        run_git(["--version"])


def test_run_git_timeout_message_is_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        # Real TimeoutExpired carries stderr as bytes even with text=True.
        raise subprocess.TimeoutExpired(cmd="git fetch", timeout=1, stderr=b"partial output")

    monkeypatch.setattr(fetch.subprocess, "run", fake_run)
    with pytest.raises(CorpusFetchError, match="timed out after 120 seconds") as info:
        run_git(["fetch"])
    assert "b'" not in str(info.value)


def test_run_git_never_prompts_for_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def fake_run(*args: object, env: dict[str, str], **kwargs: object) -> object:
        seen.update(env)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(fetch.subprocess, "run", fake_run)
    assert run_git(["status"]) == "ok"
    assert seen["GIT_TERMINAL_PROMPT"] == "0"
