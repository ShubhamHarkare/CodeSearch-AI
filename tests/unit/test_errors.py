"""Tests for codesearch/errors.py."""

import pytest

from codesearch.errors import CodeSearchError, IndexNotReadyError

ALL_ERRORS = CodeSearchError.__subclasses__()


def test_hierarchy_is_not_empty() -> None:
    assert len(ALL_ERRORS) >= 7


@pytest.mark.parametrize("error_cls", ALL_ERRORS, ids=lambda c: c.__name__)
def test_every_error_has_safe_http_fields(error_cls: type[CodeSearchError]) -> None:
    error = error_cls()
    assert isinstance(error, CodeSearchError)
    assert 400 <= error.http_status <= 599
    assert error.public_message.strip()


def test_internal_message_stays_separate_from_public_message() -> None:
    error = IndexNotReadyError("chroma collection missing at index/")
    assert "chroma collection missing" in str(error)
    assert "chroma" not in error.public_message
    assert error.http_status == 503


def test_overrides_apply_to_one_instance_only() -> None:
    error = IndexNotReadyError(public_message="Custom.", http_status=500)
    assert error.public_message == "Custom."
    assert error.http_status == 500
    assert IndexNotReadyError().http_status == 503
