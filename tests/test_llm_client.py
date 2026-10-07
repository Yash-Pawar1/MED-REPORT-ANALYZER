"""Tests for LLM Client abstraction with full mocking of Gemini and OpenAI SDKs.

Zero internet or live API key dependencies.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.llm_client import (
    GeminiClient,
    OpenAIClient,
    get_llm_client,
    LLMAuthenticationError,
    LLMEmptyResponseError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMClientError,
)


class TestLLMFactory:
    """Test provider selection factory."""

    def test_factory_returns_gemini(self):
        client = get_llm_client(provider="gemini", api_key="dummy_key")
        assert isinstance(client, GeminiClient)
        assert client.provider_name == "gemini"

    def test_factory_returns_openai(self):
        client = get_llm_client(provider="openai", api_key="dummy_key")
        assert isinstance(client, OpenAIClient)
        assert client.provider_name == "openai"

    def test_factory_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm_client(provider="unsupported_ollama_local")


class TestGeminiClient:
    """Test GeminiClient with mocked google-genai SDK."""

    def test_missing_api_key_raises_auth_error(self):
        client = GeminiClient(api_key="")
        with pytest.raises(LLMAuthenticationError, match="not configured"):
            client.generate("Test prompt")

    @patch("google.genai.Client")
    def test_successful_gemini_generation(self, mock_genai_client_cls):
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "MEDICAL DOCUMENT SUMMARY\nSymptoms:\n- Fever"
        mock_instance.models.generate_content.return_value = mock_response
        mock_genai_client_cls.return_value = mock_instance

        client = GeminiClient(api_key="valid_test_key", model="gemini-2.5-flash")
        result = client.generate("Summarize findings")
        assert "MEDICAL DOCUMENT SUMMARY" in result
        mock_instance.models.generate_content.assert_called_once()

    @patch("google.genai.Client")
    def test_gemini_empty_response_raises(self, mock_genai_client_cls):
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_instance.models.generate_content.return_value = mock_response
        mock_genai_client_cls.return_value = mock_instance

        client = GeminiClient(api_key="valid_test_key")
        with pytest.raises(LLMEmptyResponseError):
            client.generate("Prompt")

    @patch("google.genai.Client")
    def test_gemini_timeout_handling(self, mock_genai_client_cls):
        mock_instance = MagicMock()
        mock_instance.models.generate_content.side_effect = Exception("Deadline exceeded timeout")
        mock_genai_client_cls.return_value = mock_instance

        client = GeminiClient(api_key="valid_test_key")
        with pytest.raises(LLMTimeoutError):
            client.generate("Prompt")

    @patch("google.genai.Client")
    def test_gemini_rate_limit_handling(self, mock_genai_client_cls):
        mock_instance = MagicMock()
        mock_instance.models.generate_content.side_effect = Exception("Resource_exhausted quota exceeded")
        mock_genai_client_cls.return_value = mock_instance

        client = GeminiClient(api_key="valid_test_key")
        with pytest.raises(LLMRateLimitError):
            client.generate("Prompt")

    @patch("google.genai.Client")
    def test_gemini_test_connection_success(self, mock_genai_client_cls):
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "pong"
        mock_instance.models.generate_content.return_value = mock_response
        mock_genai_client_cls.return_value = mock_instance

        client = GeminiClient(api_key="valid_key")
        success, msg = client.test_connection()
        assert success is True
        assert "Successfully connected" in msg


class TestOpenAIClient:
    """Test OpenAIClient with mocked openai SDK."""

    def test_missing_api_key_raises_auth_error(self):
        client = OpenAIClient(api_key="")
        with pytest.raises(LLMAuthenticationError, match="not configured"):
            client.generate("Test prompt")

    @patch("openai.OpenAI")
    def test_successful_openai_generation(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "MEDICAL DOCUMENT SUMMARY\nSymptoms:\n- Headache"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_instance

        client = OpenAIClient(api_key="valid_key", model="gpt-4o-mini")
        result = client.generate("Test prompt")
        assert "MEDICAL DOCUMENT SUMMARY" in result
        mock_instance.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_openai_empty_response_raises(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "   "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_instance

        client = OpenAIClient(api_key="valid_key")
        with pytest.raises(LLMEmptyResponseError):
            client.generate("Prompt")

    @patch("openai.OpenAI")
    def test_openai_timeout_handling(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.side_effect = Exception("Request timeout")
        mock_openai_cls.return_value = mock_instance

        client = OpenAIClient(api_key="valid_key")
        with pytest.raises(LLMTimeoutError):
            client.generate("Prompt")

    @patch("openai.OpenAI")
    def test_openai_test_connection_success(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "pong"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_instance

        client = OpenAIClient(api_key="valid_key")
        success, msg = client.test_connection()
        assert success is True
        assert "Successfully connected" in msg
