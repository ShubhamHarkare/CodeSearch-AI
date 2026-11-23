"""FeedbackStore: one vote per answer, counted and listed (SQLite)."""

import threading
from pathlib import Path

import pytest

from codesearch.monitoring.feedback import Feedback, FeedbackStore


def test_a_new_store_is_empty() -> None:
    store = FeedbackStore()
    assert len(store) == 0
    assert store.summary() == {"up": 0, "down": 0}
    assert store.thumbs_down() == []


def test_votes_are_counted_up_and_down() -> None:
    store = FeedbackStore()
    store.add("t1", 1)
    store.add("t2", 1)
    store.add("t3", -1, "wrong hook")
    assert store.summary() == {"up": 2, "down": 1}
    assert len(store) == 3


@pytest.mark.parametrize("vote", [0, 2, -2, 5])
def test_only_plus_or_minus_one_is_a_vote(vote: int) -> None:
    store = FeedbackStore()
    with pytest.raises(ValueError):
        store.add("t1", vote)
    assert len(store) == 0  # nothing half-written


def test_voting_again_on_the_same_answer_replaces_the_vote() -> None:
    # A user who clicks 👍 then changes their mind to 👎 counts once, as 👎.
    store = FeedbackStore()
    store.add("t1", 1)
    store.add("t1", -1, "actually it's outdated")
    assert store.summary() == {"up": 0, "down": 1}
    assert len(store) == 1
    assert store.thumbs_down()[0].comment == "actually it's outdated"


def test_thumbs_down_lists_only_down_votes_newest_first() -> None:
    store = FeedbackStore()
    store.add("old", -1, "first")
    store.add("good", 1)
    store.add("new", -1)
    rows = store.thumbs_down()
    assert [r.trace_id for r in rows] == ["new", "old"]
    assert all(isinstance(r, Feedback) and r.vote == -1 for r in rows)
    assert rows[0].comment is None
    assert rows[1].comment == "first"
    assert rows[0].created_at >= rows[1].created_at > 0


def test_thumbs_down_is_capped() -> None:
    store = FeedbackStore()
    for i in range(10):
        store.add(f"t{i}", -1)
    assert len(store.thumbs_down(n=3)) == 3
    assert [r.trace_id for r in store.thumbs_down(n=3)] == ["t9", "t8", "t7"]


def test_feedback_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "feedback.db"  # the folder is created if missing
    store = FeedbackStore(path)
    store.add("t1", -1, "missing an example")
    store.close()
    reopened = FeedbackStore(path)
    assert reopened.summary() == {"up": 0, "down": 1}
    assert reopened.thumbs_down()[0].comment == "missing an example"


def test_votes_from_many_threads_are_all_kept() -> None:
    # FastAPI runs sync work in a thread pool: the store must be safe to share.
    store = FeedbackStore()

    def vote(start: int) -> None:
        for i in range(start, start + 50):
            store.add(f"t{i}", 1 if i % 2 else -1)

    threads = [threading.Thread(target=vote, args=(n * 50,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store) == 400
    assert store.summary() == {"up": 200, "down": 200}
