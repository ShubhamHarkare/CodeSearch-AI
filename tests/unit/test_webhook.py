"""Tests for the reindex webhook's HMAC signature check (codesearch/api/deps.py).

GitHub signs each delivery: X-Hub-Signature-256: sha256=<hex HMAC-SHA256(secret, body)>.
Only someone who knows the secret can make a valid signature, so only they can
trigger a rebuild.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

from codesearch.api.deps import check_signature
from codesearch.errors import InvalidWebhookError

SECRET = "s3cret"
BODY = b'{"ref": "refs/heads/main"}'


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_a_correct_signature_passes() -> None:
    check_signature(SECRET, BODY, sign(BODY))  # no exception


@pytest.mark.parametrize(
    "header",
    [
        sign(BODY, secret="wrong"),  # signed with another secret
        sign(BODY + b" "),  # the body was changed after signing
        sign(BODY)[len("sha256=") :],  # no "sha256=" prefix
        "sha1=" + hmac.new(SECRET.encode(), BODY, hashlib.sha1).hexdigest(),
        "",
        None,
    ],
)
def test_a_bad_or_missing_signature_is_rejected(header: str | None) -> None:
    with pytest.raises(InvalidWebhookError):
        check_signature(SECRET, BODY, header)


def test_the_webhook_is_off_without_a_secret() -> None:
    """With no secret configured, nothing can be verified, so nothing is accepted."""
    with pytest.raises(InvalidWebhookError) as caught:
        check_signature("", BODY, sign(BODY, secret=""))
    assert "not enabled" in caught.value.public_message
