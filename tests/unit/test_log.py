import json

import pytest
import structlog

from codesearch.log import bind_trace_id, configure_logging, get_logger, new_trace_id


@pytest.fixture(autouse=True)
def reset_log_context():
    """Ensure contextvars and log configuration are reset between tests."""
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()


def test_new_trace_id_format():
    tid = new_trace_id()
    assert isinstance(tid, str)
    assert len(tid) == 32
    # Verify it's valid hex
    int(tid, 16)
    # Verify uniqueness
    assert tid != new_trace_id()


def test_trace_id_appears_in_json_log(capsys):
    configure_logging(level="INFO", json=True)

    tid = new_trace_id()
    bind_trace_id(tid)

    logger = get_logger("test_service")
    logger.info("User logged in", user_id="usr_123")

    captured = capsys.readouterr().out.strip()
    data = json.loads(captured)

    assert data["trace_id"] == tid
    assert data["event"] == "User logged in"
    assert data["level"] == "info"
    assert data["user_id"] == "usr_123"
    assert data["logger"] == "test_service"
    assert "timestamp" in data


def test_every_line_in_a_request_carries_the_trace_id(capsys):
    configure_logging(level="INFO", json=True)
    logger = get_logger("test_service")

    tid = new_trace_id()
    bind_trace_id(tid)
    logger.info("retrieval.done", n_candidates=40)
    logger.info("generation.done", tokens=212)

    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 2
    assert all(line["trace_id"] == tid for line in lines)


def test_lowercase_level_is_accepted(capsys):
    configure_logging(level="info", json=True)
    get_logger("test_service").info("lowercase level works")

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1


def test_configure_logging_no_duplicate_lines(capsys):
    # Calling configure_logging multiple times should not cause duplicate logs
    configure_logging(level="INFO", json=True)
    configure_logging(level="INFO", json=True)

    logger = get_logger("test_service")
    logger.info("Single attempt")

    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
    assert len(lines) == 1


def test_log_level_filtering(capsys):
    configure_logging(level="WARNING", json=True)
    logger = get_logger("test_service")

    logger.debug("Debug message - should be ignored")
    logger.info("Info message - should be ignored")
    logger.warning("Warning message - should appear")

    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
    assert len(lines) == 1

    data = json.loads(lines[0])
    assert data["event"] == "Warning message - should appear"
    assert data["level"] == "warning"


def test_console_renderer_mode(capsys):
    configure_logging(level="INFO", json=False)

    logger = get_logger("test_service")
    logger.info("Console output test")

    output = capsys.readouterr().out.strip()

    # Standard JSON parsing should fail for console-rendered output
    with pytest.raises(json.JSONDecodeError):
        json.loads(output)

    assert "Console output test" in output


def test_bind_trace_id_clears_previous_state(capsys):
    configure_logging(level="INFO", json=True)
    logger = get_logger("test_service")

    bind_trace_id("first-trace-id")
    bind_trace_id("second-trace-id")

    logger.info("Testing context clear")

    data = json.loads(capsys.readouterr().out.strip())
    assert data["trace_id"] == "second-trace-id"
    assert data["logger"] == "test_service"
