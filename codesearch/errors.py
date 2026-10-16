"""One exception hierarchy for the whole package.

What goes here:
    - ``CodeSearchError(Exception)``: base class with a safe, user-facing
      ``public_message`` and an ``http_status``.
    - Subclasses: ``ConfigError``, ``CorpusFetchError``, ``IndexNotReadyError``
      (503), ``LLMUnavailableError`` (503), ``LLMTimeoutError`` (504),
      ``InvalidWebhookError`` (401), ``TraceNotFoundError`` (404).

What it must do:
    - Let api/main.py map any CodeSearchError to an HTTP response without
      leaking internal error text to users.

Must not:
    - be bypassed: raise these, never bare ``Exception``; catch broad
      exceptions only at the API boundary.

Phase: 0
"""


class CodeSearchError(Exception):
    """Base exception for all CodeSearch AI errors."""

    http_status: int = 500
    public_message: str = "An internal server error occurred."

    def __init__(
        self,
        message: str | None = None,
        public_message: str | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message or public_message or self.public_message)
        if public_message is not None:
            self.public_message = public_message
        if http_status is not None:
            self.http_status = http_status


class ConfigError(CodeSearchError):
    """Raised when application configuration is invalid or missing."""

    public_message: str = "Configuration error."


class CorpusFetchError(CodeSearchError):
    """Raised when the react.dev corpus cannot be fetched or checked out."""

    public_message: str = "Failed to fetch the documentation corpus."


class IndexNotReadyError(CodeSearchError):
    """Raised when a query arrives before the search index is loaded."""

    http_status: int = 503
    public_message: str = "The search index is still loading."


class LLMUnavailableError(CodeSearchError):
    """Raised when the LLM provider cannot be reached or keeps failing."""

    http_status: int = 503
    public_message: str = "The language model is unavailable."


class LLMTimeoutError(CodeSearchError):
    """Raised when an LLM call exceeds the allowed time limit."""

    http_status: int = 504
    public_message: str = "LLM request timed out."


class InvalidWebhookError(CodeSearchError):
    """Raised when webhook authentication fails."""

    http_status: int = 401
    public_message: str = "Invalid or missing webhook signature."


class TraceNotFoundError(CodeSearchError):
    """Raised when an execution trace ID is not found."""

    http_status: int = 404
    public_message: str = "Requested trace ID was not found."
