"""HTTP client the Streamlit UI uses to talk to the API.

``ApiClient`` wraps every endpoint the UI calls: ask, ask_stream (server-sent
events), explain, deep_explain, ready, index_info. Every failure (API down, timeout,
HTTP error, an ``error`` event mid-stream) becomes ``ApiError`` with a message fit
to show users, so the UI never crashes on a network problem.

It imports nothing from codesearch except schemas.py: the UI stays a pure HTTP
client (tests/unit/test_api_client.py checks this).
"""

import json
from collections.abc import Iterator
from types import TracebackType
from typing import Any

import httpx

from codesearch.schemas import StreamEvent


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

    def explain(self, trace_id: str) -> dict[str, Any]:
        return self._handle_response(self._request("GET", f"/v1/explain/{trace_id}"))

    def deep_explain(self, trace_id: str) -> dict[str, Any]:
        """POST /v1/explain/{trace_id}/deep -> {"trace_id", "scores"} (slow: k LLM calls).

        409 (cache hit, abstain) raises ApiError like any other error.
        """
        return self._handle_response(self._request("POST", f"/v1/explain/{trace_id}/deep"))

    def ask_stream(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        *,
        bypass_cache: bool = False,
    ) -> Iterator[StreamEvent]:
        """POST /v1/ask/stream -> events as they arrive: retrieval, token..., done.

        An ``error`` event, an HTTP error, a dropped connection, or a stream that ends
        without ``done`` all raise ApiError, after yielding whatever arrived before.
        """
        payload = {"question": question, "history": history or [], "bypass_cache": bypass_cache}
        try:
            with self._http.stream("POST", "/v1/ask/stream", json=payload) as response:
                if not response.is_success:
                    response.read()  # a streamed response must be read before .json()
                    self._handle_response(response)  # raises ApiError
                trace_id = response.headers.get("X-Trace-Id")
                for event in parse_sse(response.iter_lines()):
                    if event.type == "error":
                        data = event.data if isinstance(event.data, dict) else {}
                        raise ApiError(
                            data.get("error") or "The answer failed part-way through.",
                            trace_id=data.get("trace_id") or trace_id,
                        )
                    yield event
                    if event.type == "done":
                        return
        except httpx.TimeoutException as e:
            raise ApiError(f"The API stopped responding for {self.timeout_s:g}s.") from e
        except httpx.HTTPError as e:
            raise ApiError(
                f"Lost the connection to the API at {self.base_url}. Is `make serve` running?"
            ) from e
        except ValueError as e:  # bad JSON or an unknown event type
            raise ApiError("The API sent an event this app can't read.") from e
        raise ApiError("The answer stream ended before the answer was complete.")

    def send_feedback(self, trace_id: str, vote: int, comment: str | None = None) -> None:
        """POST /v1/feedback. 204 has no body, so only an error response is parsed.

        Voting again on the same trace_id replaces the earlier vote.
        """
        payload: dict[str, Any] = {"trace_id": trace_id, "vote": vote}
        if comment:
            payload["comment"] = comment
        response = self._request("POST", "/v1/feedback", json=payload)
        if not response.is_success:
            self._handle_response(response)  # raises ApiError

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


def parse_sse(lines: Iterator[str]) -> Iterator[StreamEvent]:
    """Server-sent event lines -> StreamEvents. A blank line ends each event.

    Only ``event:`` and ``data:`` fields are used; comments (``: ping``) and other
    fields are skipped. ``data`` is JSON, so a token containing a newline arrives
    escaped on one line.
    """
    kind: str | None = None
    data: list[str] = []
    for line in lines:
        if line == "":
            if kind is not None and data:
                yield StreamEvent.model_validate(
                    {"type": kind, "data": json.loads("\n".join(data))}
                )
            kind, data = None, []
        elif line.startswith("event:"):
            kind = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].lstrip())
    if kind is not None and data:  # a last event without its closing blank line
        yield StreamEvent.model_validate({"type": kind, "data": json.loads("\n".join(data))})
