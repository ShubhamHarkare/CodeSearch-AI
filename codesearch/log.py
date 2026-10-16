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

import logging
import uuid

import structlog


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # adds trace_id (from bind_trace_id)
            structlog.processors.add_log_level,  # adds "level": "info"
            structlog.processors.TimeStamper(fmt="iso"),  # adds "timestamp": "2026-10-16T..."
            renderer,  # dict -> JSON text (must be last)
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(),  # write to stdout
        cache_logger_on_first_use=False,  # so calling configure again takes effect
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger().bind(logger=name)
    return logger


def new_trace_id() -> str:
    return uuid.uuid4().hex


def bind_trace_id(trace_id: str) -> None:
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
