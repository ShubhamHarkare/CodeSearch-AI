"""LLM clients implementing the LLM protocol. Everything runs locally through Ollama.

What goes here:
    - ``OllamaLLM(model="gemma3:4b", base_url="http://localhost:11434", ...)``:
        * ``async generate(prompt, system=None) -> str`` via POST /api/generate
          (max_tokens is Ollama's ``num_predict``)
        * ``async stream(prompt, system=None) -> AsyncIterator[str]`` (Phase 3):
          yields tokens; closes the upstream response if the consumer stops
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

from collections.abc import AsyncIterator
from typing import Any

import httpx

from codesearch.errors import LLMTimeoutError, LLMUnavailableError


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

    async def stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
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

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """One complete answer from POST /api/generate (no streaming)."""
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # one JSON reply instead of one line per token
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        if system:
            body["system"] = system

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

    async def stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        """Phase 3: token streaming with cancellation."""
        raise NotImplementedError("OllamaLLM.stream comes in Phase 3")
        yield ""  # makes this an async generator, so the type matches the protocol
