"""Phase 1 tests for the real LLM clients (GroqLLM, OllamaLLM).

Skipped until those classes exist in codesearch/generation/llm.py. When you write
them, delete the skip line below.
"""

import pytest

pytest.skip("Phase 1: GroqLLM / OllamaLLM not written yet", allow_module_level=True)

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

import groq  # noqa: E402

from codesearch.errors import LLMTimeoutError, LLMUnavailableError  # noqa: E402
from codesearch.generation.llm import GroqLLM, OllamaLLM  # noqa: E402

# =====================================================================
# GroqLLM Tests
# =====================================================================


@pytest.fixture
def mock_groq_client():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_groq_generate_success(mock_groq_client):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Groq response text"))]
    mock_groq_client.chat.completions.create.return_value = mock_response

    llm = GroqLLM(model="llama-3.3-70b-versatile", api_key="fake-key")
    llm._client = mock_groq_client  # inject mock client

    result = await llm.generate("Hello")

    assert result == "Groq response text"
    mock_groq_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_groq_stream_success(mock_groq_client):
    async def mock_stream_chunks():
        chunks = ["Hello", " ", "world", "!"]
        for chunk in chunks:
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock(delta=MagicMock(content=chunk))]
            yield mock_chunk

    mock_groq_client.chat.completions.create.return_value = mock_stream_chunks()

    llm = GroqLLM(model="llama-3.3-70b-versatile", api_key="fake-key")
    llm._client = mock_groq_client

    tokens = [token async for token in llm.stream("Hello")]
    assert "".join(tokens) == "Hello world!"


@pytest.mark.asyncio
async def test_groq_stream_client_disconnect_closes_upstream(mock_groq_client):
    """Verify that early exit closes the upstream stream iterator."""
    upstream_closed = False

    class MockAsyncStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock(delta=MagicMock(content="token"))]
            return mock_chunk

        async def close(self):
            nonlocal upstream_closed
            upstream_closed = True

    mock_groq_client.chat.completions.create.return_value = MockAsyncStream()

    llm = GroqLLM(model="llama-3.3-70b-versatile", api_key="fake-key")
    llm._client = mock_groq_client

    # Consume only 1 token and break
    async for token in llm.stream("Hello"):
        assert token == "token"
        break

    assert upstream_closed, "Upstream stream was not closed upon consumer disconnect"


@pytest.mark.asyncio
async def test_groq_retry_on_rate_limit(mock_groq_client):
    """Verify backoff retries on rate limit (429) then succeeds."""
    rate_limit_error = groq.RateLimitError(
        message="Rate limit hit", response=MagicMock(status_code=429), body=None
    )

    mock_success = MagicMock()
    mock_success.choices = [MagicMock(message=MagicMock(content="Success after retry"))]

    mock_groq_client.chat.completions.create.side_effect = [rate_limit_error, mock_success]

    llm = GroqLLM(model="llama-3.3-70b-versatile", api_key="fake-key", max_retries=2)
    llm._client = mock_groq_client

    result = await llm.generate("Test prompt")

    assert result == "Success after retry"
    assert mock_groq_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_groq_raises_unavailable_error_on_max_retries(mock_groq_client):
    server_error = groq.InternalServerError(
        message="Groq 500 error", response=MagicMock(status_code=500), body=None
    )
    mock_groq_client.chat.completions.create.side_effect = server_error

    llm = GroqLLM(model="llama-3.3-70b-versatile", api_key="fake-key", max_retries=2)
    llm._client = mock_groq_client

    with pytest.raises(LLMUnavailableError):
        await llm.generate("Test prompt")


@pytest.mark.asyncio
async def test_groq_raises_timeout_error(mock_groq_client):
    mock_groq_client.chat.completions.create.side_effect = groq.APITimeoutError(request=MagicMock())

    llm = GroqLLM(model="llama-3.3-70b-versatile", api_key="fake-key")
    llm._client = mock_groq_client

    with pytest.raises(LLMTimeoutError):
        await llm.generate("Test prompt")


# =====================================================================
# OllamaLLM Tests
# =====================================================================


@pytest.mark.asyncio
async def test_ollama_num_predict_mapping():
    """Verify max_tokens is correctly mapped to num_predict for Ollama API."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Local response"}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        llm = OllamaLLM(model="gemma3:4b", base_url="http://localhost:11434", max_tokens=256)
        res = await llm.generate("Local prompt")

        assert res == "Local response"

        # Verify request body contains options.num_predict
        _, kwargs = mock_post.call_args
        json_body = kwargs["json"]
        assert json_body["options"]["num_predict"] == 256
        assert json_body["model"] == "gemma3:4b"


@pytest.mark.asyncio
async def test_ollama_raises_unavailable_error_on_connection_failure():
    with patch("httpx.AsyncClient.post", side_effect=Exception("Connection refused")):
        llm = OllamaLLM(model="gemma3:4b", base_url="http://localhost:11434")

        with pytest.raises(LLMUnavailableError):
            await llm.generate("Local prompt")
