"""Rolling in-memory statistics over the most recent requests.

What goes here:
    - ``RollingWindow(maxlen=500)`` holding RequestSignals (schemas.py):
      ``record(signals)``, ``values(field) -> list``, ``mean(field)``,
      ``percentile(field, p)``, ``rate(predicate)`` (e.g. abstain rate).
    - Thread-safe appends (deque with maxlen is enough; copy before reading).

Tested by: tests/unit/test_windows.py (maxlen respected, p50/p95 on known data).
Phase: 3
"""
