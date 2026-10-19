"""LLM clients implementing the LLM protocol. The only module that imports groq.

What goes here:
    - ``GroqLLM(model, api_key, temperature, max_tokens, timeout_s, max_retries)``:
        * ``async generate(prompt) -> str``
        * ``async stream(prompt) -> AsyncIterator[str]`` yielding tokens; closes
          the upstream stream if the consumer stops (client disconnected)
        * retries with backoff on rate limits / 5xx (tenacity), raises
          LLMTimeoutError / LLMUnavailableError from errors.py
    - ``OllamaLLM(model, base_url, ...)``: same interface, for local runs
      (you currently use gemma3:4b). Use Ollama's ``num_predict`` for max tokens.
    - ``FakeLLM(responses=None)``: deterministic, no network; by default returns
      an answer that cites [1] and [2] so citation and groundedness code is
      exercised in tests. Records the prompts it received.

Tested by: tests/unit/test_llm.py (FakeLLM streaming yields the full text;
GroqLLM retry logic with a mocked client).
Phase: 1 (generate), 3 (stream + cancellation)
"""

from collections.abc import AsyncIterator


class FakeLLM:
    """Deterministic stand-in for a real LLM: no network, records every prompt."""

    DEFAULT_ANSWER = (
        "Use useState to add state to a component [1]. It returns a value and a setter [2]."
    )

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses: list[str] = list(responses or [])
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)  # canned answers, in order
        return self.DEFAULT_ANSWER

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        answer = await self.generate(prompt)  # records the prompt too
        words = answer.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "
