"""Structured JSON logging. Named log.py so it never shadows stdlib ``logging``.

What goes here:
    - ``configure_logging(level: str, json: bool = True) -> None``: set up
      structlog once (JSON renderer in prod, console renderer locally),
      ISO timestamps, log level, logger name.
    - ``get_logger(name: str)``: return a bound structlog logger.
    - ``bind_trace_id(trace_id: str)`` / ``new_trace_id() -> str`` using
      structlog contextvars, so every line in one request carries its trace_id.

What it must do:
    - Be safe to call configure_logging() more than once (no duplicate lines;
      v1's logger.py wrote lines twice).

Must not:
    - write files itself (containers log to stdout), or log secrets or full
      prompts at INFO level.

Tested by: tests/unit/test_log.py (trace_id appears on every line).
Phase: 0
"""
