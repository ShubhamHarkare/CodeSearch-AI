"""Rolling statistics over the most recent answers, for the /metrics gauges.

RollingWindow keeps the last N RequestSignals (one per answer) in memory, so
/metrics can report e.g. "abstain rate over the last 500 answers". Statistics
over an empty window are None ("no data yet"), never 0.
"""

import math
import threading
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime

from codesearch.schemas import Answer, RequestSignals


def signals_from_answer(answer: Answer, question: str, retrieval_mode: str) -> RequestSignals:
    """The monitoring view of one answer."""
    return RequestSignals(
        trace_id=answer.trace_id,
        timestamp=datetime.now(UTC),
        query=question,
        retrieval_mode=retrieval_mode,
        latency_ms=answer.timings.get("total_ms", 0.0),
        cache_tier=answer.cache_tier,
        abstained=answer.abstained,
        groundedness=answer.groundedness,
        confidence=answer.confidence,
        ttft_ms=answer.timings.get("ttft_ms"),  # None unless streamed
        retrieval_ms=answer.timings.get("retrieval_ms"),  # None on a cache hit
    )


class RollingWindow:
    """The last `maxlen` answers' signals; older ones drop off automatically."""

    def __init__(self, maxlen: int = 500) -> None:
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        self._items: deque[RequestSignals] = deque(maxlen=maxlen)  # by name: it's the size
        self._lock = threading.Lock()

    def record(self, signals: RequestSignals) -> None:
        with self._lock:
            self._items.append(signals)

    def snapshot(self) -> list[RequestSignals]:
        """A copy, so statistics can loop over it while others keep recording."""
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def values(self, field: str) -> list[float]:
        """Every non-None value of `field`, as floats, oldest first."""
        result: list[float] = []
        for signals in self.snapshot():
            value = getattr(signals, field)  # e.g. signals.groundedness
            if value is not None:  # cache hits / abstains have no groundedness
                result.append(float(value))
        return result

    def mean(self, field: str) -> float | None:
        values = self.values(field)
        if not values:  # an empty list, not None
            return None
        return sum(values) / len(values)

    def percentile(self, field: str, p: float) -> float | None:
        """Nearest-rank percentile, like eval/loadtest.py: p50 of 1..100 is 50."""
        if not 0 <= p <= 100:
            raise ValueError(f"p must be in [0, 100], got {p}")
        values = sorted(self.values(field))
        if not values:
            return None
        rank = max(1, math.ceil(p / 100 * len(values)))
        return values[rank - 1]

    def rate(self, predicate: Callable[[RequestSignals], bool]) -> float | None:
        """The share of answers for which predicate(signals) is True; None if empty."""
        items = self.snapshot()  # one copy for both counts
        if not items:
            return None
        return sum(1 for s in items if predicate(s)) / len(items)
