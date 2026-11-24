"""LLM clients implementing the LLM protocol. Everything runs locally through Ollama.

What goes here:
    - ``OllamaLLM(model="gemma3:4b", base_url="http://localhost:11434", ...)``:
        * ``async generate(prompt, system=None) -> str`` via POST /api/generate
          (max_tokens is Ollama's ``num_predict``)
        * ``async stream(prompt, system=None) -> AsyncGenerator[str]``: yields text
          as it is generated; closing the iterator closes the upstream response
        * raises LLMTimeoutError / LLMUnavailableError from errors.py
    - ``FakeLLM(responses=None)``: deterministic, no network; by default returns
      an answer that cites [1] and [2] so citation and groundedness code is
      exercised in tests. Records the prompts and system prompts it received.

Setup for real answers: ``ollama serve`` and ``ollama pull gemma3:4b``, then set
CODESEARCH_LLM__PROVIDER=ollama in .env.

Tested by: tests/unit/test_llm.py (FakeLLM) and tests/unit/test_llm_providers.py
(OllamaLLM against httpx.MockTransport; one needs_llm test against real Ollama).
Phase: 1 (generate), 3 (stream + cancellation)
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from codesearch.errors import LLMRateLimitedError, LLMTimeoutError, LLMUnavailableError


class FakeLLM:
    """Deterministic stand-in for a real LLM: no network, records every prompt."""

    DEFAULT_ANSWER = (
        "Use useState to add state to a component [1]. It returns a value and a setter [2]."
    )

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses: list[str] = list(responses or [])
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        if self.responses:
            return self.responses.pop(0)  # canned answers, in order
        return self.DEFAULT_ANSWER

    async def stream(self, prompt: str, system: str | None = None) -> AsyncGenerator[str]:
        answer = await self.generate(prompt, system)  # records the prompt too
        words = answer.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "


class OllamaLLM:
    """A local model served by Ollama (https://ollama.com): free, no API key."""

    def __init__(
        self,
        model: str = "gemma3:4b",
        base_url: str = "http://localhost:11434",
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._client = client  # tests inject one backed by httpx.MockTransport

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()  # created on first use, then reused
        return self._client

    def _body(self, prompt: str, system: str | None, *, stream: bool) -> dict[str, Any]:
        """The /api/generate request. stream=False: one JSON reply; True: one line per token."""
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        if system:
            body["system"] = system
        return body

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """One complete answer from POST /api/generate (no streaming)."""
        body = self._body(prompt, system, stream=False)
        url = f"{self.base_url}/api/generate"
        try:
            response = await self._http().post(url, json=body, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as error:  # check before HTTPError: it's a subclass
            raise LLMTimeoutError(
                f"Ollama did not answer within {self.timeout_s}s ({self.model})"
            ) from error
        except httpx.HTTPError as error:
            raise LLMUnavailableError(
                f"Ollama request failed ({self.model} at {self.base_url}): {error}. "
                f"Is `ollama serve` running and the model pulled (`ollama pull {self.model}`)?"
            ) from error

        if not isinstance(data, dict) or "response" not in data:
            raise LLMUnavailableError(f"Unexpected reply from Ollama: {str(data)[:200]}")
        return str(data["response"])

    async def stream(self, prompt: str, system: str | None = None) -> AsyncGenerator[str]:
        """Yield the answer as Ollama produces it; closing the iterator closes the connection.

        Ollama sends one JSON object per line ({"response": "<text>", "done": false}),
        ending with {"done": true}. If the consumer stops early (the user left), Python
        closes this generator at its ``yield``: the ``async with`` then closes the HTTP
        response, which tells Ollama to stop generating.
        """
        body = self._body(prompt, system, stream=True)
        url = f"{self.base_url}/api/generate"
        try:
            async with self._http().stream(
                "POST", url, json=body, timeout=self.timeout_s
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise LLMUnavailableError(
                            f"Malformed line from Ollama: {line[:200]!r}"
                        ) from e

                    if "error" in data:
                        raise LLMUnavailableError(f"Ollama error ({self.model}): {data['error']}")

                    if chunk := data.get("response"):
                        yield str(chunk)

                    if data.get("done"):
                        return
        except httpx.TimeoutException as e:  # check before HTTPError: it's a subclass
            raise LLMTimeoutError(
                f"Ollama stopped sending within {self.timeout_s}s ({self.model})"
            ) from e
        except httpx.HTTPError as e:
            raise LLMUnavailableError(
                f"Ollama request failed ({self.model} at {self.base_url}): {e}. "
                f"Is `ollama serve` running and the model pulled (`ollama pull {self.model}`)?"
            ) from e


def _error_code(response: httpx.Response) -> str:
    """Groq's short error code, e.g. " (model_not_found)"; never the free-text message.

    The code is a fixed word, so it's safe to log; the message can echo user input or the key.
    """
    try:
        code = response.json().get("error", {}).get("code")
    except (ValueError, AttributeError):
        return ""
    return f" ({code})" if isinstance(code, str) and code.isidentifier() else ""


class OpenAICompatLLM:
    """Any server speaking the OpenAI chat-completions API (Groq, OpenRouter, vLLM, ...)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("an API key is required (set CODESEARCH_LLM__API_KEY)")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._key = api_key
        self._client = client
        self.reasoning_effort = reasoning_effort

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()  # created on first use, then reused
        return self._client

    def _body(self, prompt: str, system: str | None, *, stream: bool) -> dict[str, Any]:
        """The chat-completions request: the rules as the system message, then the prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def _check(self, response: httpx.Response) -> None:
        """Raise the matching error for a non-2xx response (never the body: it can echo the key)."""
        if response.status_code == 429:
            wait = response.headers.get("retry-after")  # e.g. "7", or None if Groq didn't send it
            message = f"{self.model}: rate limited"
            if wait:
                message += f", retry after {wait}s"
            raise LLMRateLimitedError(message)
        if response.status_code in (401, 403):
            raise LLMUnavailableError(
                f"{self.model}: the API key was rejected{_error_code(response)}"
            )
        if not response.is_success:
            raise LLMUnavailableError(
                f"{self.model}: HTTP {response.status_code}{_error_code(response)}"
            )

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """One complete answer from POST /chat/completions (no streaming)."""
        body = self._body(prompt, system, stream=False)
        try:
            response = await self._http().post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException as e:  # check before HTTPError: it's a subclass
            raise LLMTimeoutError(f"{self.model} did not answer within {self.timeout_s}s") from e
        except httpx.HTTPError as e:
            raise LLMUnavailableError(f"{self.model} at {self.base_url}: {type(e).__name__}") from e
        self._check(response)

        try:
            # content can be null: `or ""` inside str(), so None doesn't become "None"
            return str(response.json()["choices"][0]["message"]["content"] or "")
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMUnavailableError(f"{self.model}: unexpected reply shape") from e

    async def stream(self, prompt: str, system: str | None = None) -> AsyncGenerator[str]:
        """Yield the answer as Groq produces it; closing the iterator closes the connection."""
        body = self._body(prompt, system, stream=True)
        try:
            async with self._http().stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=self.timeout_s,
            ) as response:
                if not response.is_success:
                    await response.aread()  # a streamed body must be read before .json()
                self._check(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError as e:
                        raise LLMUnavailableError(f"{self.model}: malformed stream line") from e
                    if "error" in data:
                        raise LLMUnavailableError(f"{self.model}: error mid-stream")
                    choices = data.get("choices") or [{}]
                    text = (choices[0].get("delta") or {}).get("content")
                    if text:
                        yield str(text)
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"{self.model} stopped sending within {self.timeout_s}s") from e
        except httpx.HTTPError as e:
            raise LLMUnavailableError(f"{self.model} at {self.base_url}: {type(e).__name__}") from e
