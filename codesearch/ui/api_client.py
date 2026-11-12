"""HTTP client the Streamlit UI uses to talk to the API.

What goes here:
    - ``ApiClient(base_url, timeout_s)`` (httpx):
        * ``ask(question, history) -> dict``
        * ``ask_stream(question, history) -> Iterator[StreamEvent]``: parse SSE
          frames into typed events
        * ``explain(trace_id)``, ``deep_explain(trace_id)``
        * ``send_feedback(trace_id, vote, comment)``
        * ``ready() -> dict`` and ``metrics_summary() -> dict`` for the sidebar
    - Clear error messages when the API is down (show them in the UI, don't crash).

Must not:
    - import anything from codesearch except schemas.py.

Tested by: tests/unit/test_api_client.py with httpx.MockTransport.
Phase: 1, 3 (stream)
"""

from types import TracebackType
from typing import Any

import httpx


class ApiError(Exception):
    """Anything that went wrong talking to the API, with a message fit to show users."""

    def __init__(
        self, message: str, status: int | None = None, trace_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status  # HTTP status, or None if no response arrived at all
        self.trace_id = trace_id  # quote this when reporting a problem


class ApiClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout_s: float = 120.0,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")  # rstrip: only the trailing slash
        self.timeout_s = timeout_s
        # One client, reused for every call (keeps the connection open).
        # Tests pass a MockTransport; in real use transport is None.
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout_s, transport=transport)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Send one request; a failure to get any response becomes ApiError(status=None)."""
        try:
            return self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as e:  # before HTTPError: it's a subclass of it
            raise ApiError(f"The API timed out after {self.timeout_s:g}s.") from e
        except httpx.HTTPError as e:  # connection refused, DNS failure, ...
            raise ApiError(
                f"Can't reach the API at {self.base_url}. Is `make serve` running?"
            ) from e

    def _handle_response(
        self, response: httpx.Response, raise_for_status: bool = True
    ) -> dict[str, Any]:
        """The JSON body; for a non-2xx response, raise ApiError with a readable message."""
        try:
            data = response.json()
        except ValueError:  # not JSON, e.g. an HTML 502 page from a proxy
            data = None

        if raise_for_status and not response.is_success:
            status = response.status_code
            if not isinstance(data, dict):
                raise ApiError(f"The API returned an error ({status}).", status=status)
            if status == 422 and data.get("detail"):  # FastAPI validation error
                first = data["detail"][0]
                raise ApiError(f"Invalid request: {first.get('msg', 'bad input')}", status=status)
            raise ApiError(
                data.get("error") or f"The API returned an error ({status}).",
                status=status,
                trace_id=data.get("trace_id") or response.headers.get("X-Trace-Id"),
            )

        if not isinstance(data, dict):
            raise ApiError("The API returned a response that isn't JSON.", response.status_code)
        return data

    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        *,
        bypass_cache: bool = False,
    ) -> dict[str, Any]:
        """POST /v1/ask -> the AskResponse body."""
        payload = {"question": question, "history": history or [], "bypass_cache": bypass_cache}
        return self._handle_response(self._request("POST", "/v1/ask", json=payload))

    def ready(self) -> dict[str, Any]:
        """GET /readyz. A 503 ("not ready yet") is returned as data, not raised."""
        return self._handle_response(self._request("GET", "/readyz"), raise_for_status=False)

    def index_info(self) -> dict[str, Any]:
        """GET /v1/index -> manifest summary (503 before the index loads raises)."""
        return self._handle_response(self._request("GET", "/v1/index"))

    # Phase 2-3 (added with their endpoints): ask_stream, explain, deep_explain,
    # send_feedback, metrics_summary.

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ApiClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
