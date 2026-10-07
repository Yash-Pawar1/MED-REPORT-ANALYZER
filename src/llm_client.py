"""Unified LLM Provider Abstraction supporting Google Gemini and OpenAI.

Implements clean provider polymorphism with custom exception wrapping,
strict timeouts, and safe connection test probes.
"""

from abc import ABC, abstractmethod
import os
from typing import Optional, Tuple

from src.config import config
from src.utils import setup_logger

logger = setup_logger(__name__)


class LLMClientError(Exception):
    """Base exception for all LLM client errors."""
    pass


class LLMAuthenticationError(LLMClientError):
    """Raised when API key is missing, invalid, or unauthorized."""
    pass


class LLMTimeoutError(LLMClientError):
    """Raised when an API request exceeds the configured timeout."""
    pass


class LLMRateLimitError(LLMClientError):
    """Raised when quota is exceeded or rate limit is encountered."""
    pass


class LLMEmptyResponseError(LLMClientError):
    """Raised when the provider returns an empty or whitespace response."""
    pass


class BaseLLMClient(ABC):
    """Abstract Base Class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to the model and return the generated text response."""
        pass

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Verify API credentials and connectivity safely without sending medical data."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the lowercase name of the provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model name."""
        pass


class GeminiClient(BaseLLMClient):
    """Google Gemini API client using the official google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        self.api_key = (api_key or config.gemini_api_key or "").strip()
        self.model = (model or config.gemini_model or "gemini-2.5-flash").strip()
        self.timeout = timeout or config.llm_timeout
        self.temperature = temperature if temperature is not None else config.temperature
        self._client = None

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self.model

    def _get_client(self):
        """Lazy initialization of google-genai client."""
        if not self.api_key:
            raise LLMAuthenticationError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file or environment."
            )
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize google-genai Client: {e}")
                raise LLMClientError(f"Could not initialize Gemini client: {str(e)}")
        return self._client

    def generate(self, prompt: str) -> str:
        """Generate summary using Gemini API."""
        if not prompt or not prompt.strip():
            raise LLMClientError("Cannot generate response for empty prompt.")

        client = self._get_client()

        try:
            from google.genai import types

            config_params = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=config.max_output_tokens
            )

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config_params
            )

            if not response or not response.text:
                raise LLMEmptyResponseError("Gemini returned an empty response.")

            return response.text.strip()

        except LLMAuthenticationError:
            raise
        except LLMEmptyResponseError:
            raise
        except Exception as e:
            err_str = str(e).lower()
            logger.error(f"Gemini API call failed: {e}")
            if "api_key" in err_str or "unauthenticated" in err_str or "permission_denied" in err_str or "invalid" in err_str and "key" in err_str:
                raise LLMAuthenticationError(f"Gemini authentication failed: {str(e)}")
            elif "resource_exhausted" in err_str or "quota" in err_str or "rate" in err_str:
                raise LLMRateLimitError(f"Gemini rate limit or quota exceeded: {str(e)}")
            elif "deadline" in err_str or "timeout" in err_str:
                raise LLMTimeoutError(f"Gemini request timed out: {str(e)}")
            else:
                raise LLMClientError(f"Gemini generation error: {str(e)}")

    def test_connection(self) -> Tuple[bool, str]:
        """Test API connection with a safe minimal ping."""
        if not self.api_key:
            return False, "GEMINI_API_KEY is not set. Please add it to your .env file."
        try:
            client = self._get_client()
            # Test with a non-medical 1-word prompt
            from google.genai import types
            test_config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=5
            )
            resp = client.models.generate_content(
                model=self.model,
                contents="ping",
                config=test_config
            )
            if resp and resp.text:
                return True, f"Successfully connected to Gemini API (Model: {self.model})."
            return False, "Gemini responded with empty content."
        except Exception as e:
            return False, f"Connection failed: {str(e)}"


class OpenAIClient(BaseLLMClient):
    """OpenAI API client using the official openai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        self.api_key = (api_key or config.openai_api_key or "").strip()
        self.model = (model or config.openai_model or "gpt-4o-mini").strip()
        self.timeout = timeout or config.llm_timeout
        self.temperature = temperature if temperature is not None else config.temperature
        self._client = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self.model

    def _get_client(self):
        """Lazy initialization of openai.OpenAI client."""
        if not self.api_key:
            raise LLMAuthenticationError(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env file or environment."
            )
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, timeout=float(self.timeout))
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise LLMClientError(f"Could not initialize OpenAI client: {str(e)}")
        return self._client

    def generate(self, prompt: str) -> str:
        """Generate summary using OpenAI API."""
        if not prompt or not prompt.strip():
            raise LLMClientError("Cannot generate response for empty prompt.")

        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=config.max_output_tokens,
                timeout=float(self.timeout)
            )

            if not response or not response.choices:
                raise LLMEmptyResponseError("OpenAI returned an empty choice list.")

            content = response.choices[0].message.content
            if not content or not content.strip():
                raise LLMEmptyResponseError("OpenAI returned an empty response body.")

            return content.strip()

        except LLMAuthenticationError:
            raise
        except LLMEmptyResponseError:
            raise
        except Exception as e:
            err_str = str(e).lower()
            logger.error(f"OpenAI API call failed: {e}")
            if "authentication" in err_str or "api_key" in err_str or "unauthorized" in err_str:
                raise LLMAuthenticationError(f"OpenAI authentication failed: {str(e)}")
            elif "rate_limit" in err_str or "quota" in err_str:
                raise LLMRateLimitError(f"OpenAI rate limit or quota exceeded: {str(e)}")
            elif "timeout" in err_str:
                raise LLMTimeoutError(f"OpenAI request timed out: {str(e)}")
            else:
                raise LLMClientError(f"OpenAI generation error: {str(e)}")

    def test_connection(self) -> Tuple[bool, str]:
        """Test API connection with a safe minimal ping."""
        if not self.api_key:
            return False, "OPENAI_API_KEY is not set. Please add it to your .env file."
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                timeout=float(self.timeout)
            )
            if resp and resp.choices and resp.choices[0].message.content:
                return True, f"Successfully connected to OpenAI API (Model: {self.model})."
            return False, "OpenAI responded with empty content."
        except Exception as e:
            return False, f"Connection failed: {str(e)}"


def get_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
    temperature: Optional[float] = None
) -> BaseLLMClient:
    """Factory function to instantiate the selected LLM provider client.
    
    Args:
        provider: 'gemini' or 'openai' (defaults to config.llm_provider)
        api_key: Optional API key override
        model: Optional model override
        timeout: Optional timeout override
        temperature: Optional temperature override
        
    Returns:
        BaseLLMClient instance
        
    Raises:
        ValueError: If provider is unrecognized
    """
    selected_provider = (provider or config.llm_provider).lower().strip()

    if selected_provider == "gemini":
        return GeminiClient(
            api_key=api_key,
            model=model,
            timeout=timeout,
            temperature=temperature
        )
    elif selected_provider == "openai":
        return OpenAIClient(
            api_key=api_key,
            model=model,
            timeout=timeout,
            temperature=temperature
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider '{selected_provider}'. Supported options are 'gemini' and 'openai'."
        )
