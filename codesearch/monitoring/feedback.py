"""Thumbs up / down on answers, stored in SQLite next to the traces."""

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Feedback:
    trace_id: str
    vote: int
    comment: str | None
    created_at: float


class FeedbackStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()

        with self._lock, self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS feedback ("
                "trace_id TEXT PRIMARY KEY, "
                "vote INTEGER NOT NULL, "
                "comment TEXT, "
                "created_at REAL NOT NULL)"
            )

    def add(self, trace_id: str, vote: int, comment: str | None = None) -> None:
        if vote not in (1, -1):
            raise ValueError(f"vote must be 1 or -1, got {vote}")

        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO feedback (trace_id, vote, comment, created_at) "
                "VALUES (?, ?, ?, ?)",
                (trace_id, vote, comment, time.time()),
            )

    def summary(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute("SELECT vote, COUNT(*) FROM feedback GROUP BY vote").fetchall()
        votes = dict(rows)  # [(-1, 1), (1, 2)] -> {-1: 1, 1: 2}
        return {"up": votes.get(1, 0), "down": votes.get(-1, 0)}

    def thumbs_down(self, n: int = 50) -> list[Feedback]:
        with self._lock:
            rows = self._db.execute(
                "SELECT trace_id, vote, comment, created_at FROM feedback "
                "WHERE vote = -1 "
                # newest first; rowid (SQLite's hidden insert counter) breaks ties
                # when two votes have the same timestamp
                "ORDER BY created_at DESC, rowid DESC "
                "LIMIT ?",
                (n,),  # a one-element tuple needs the trailing comma
            ).fetchall()
        # Each row is (trace_id, vote, comment, created_at), the same order as Feedback's fields.
        return [Feedback(*row) for row in rows]

    def __len__(self) -> int:
        with self._lock:
            (count,) = self._db.execute("SELECT COUNT(*) FROM feedback").fetchone()
        return int(count)

    def close(self) -> None:
        self._db.close()
