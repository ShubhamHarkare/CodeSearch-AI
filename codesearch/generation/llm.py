"""LLM clients implementing the LLM protocol. The only module that imports groq.

What goes here:
    - ``GroqLLM(model, api_key, temperature, max_tokens, timeout_s, max_retries)``:
        * ``async generate(prompt) -> str``
        * ``async stream(prompt) -> AsyncIterator[str]`` yielding tokens; closes
          the upstream stream if the consumer stops (client disconnected)
        * retries with backoff on rate limits / 5xx (tenacity), raises
          LLMTimeout / LLMUnavailable from errors.py
    - ``OllamaLLM(model, base_url, ...)``: same interface, for local runs
      (you currently use gemma3:4b). Use Ollama's ``num_predict`` for max tokens.
    - ``FakeLLM(responses=None)``: deterministic, no network; by default returns
      an answer that cites [1] and [2] so citation and groundedness code is
      exercised in tests. Records the prompts it received.

Tested by: tests/unit/test_llm.py (FakeLLM streaming yields the full text;
GroqLLM retry logic with a mocked client).
Phase: 1 (generate), 3 (stream + cancellation)
"""
