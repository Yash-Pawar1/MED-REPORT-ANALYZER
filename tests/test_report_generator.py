"""Tests for LLM report generation and automatic fallback switching."""

from unittest.mock import MagicMock
import pytest

from src.llm_report_generator import LLMReportGenerator
from src.llm_client import BaseLLMClient, LLMTimeoutError
from src.models import ExtractedMedicalData, MedicalEntity, EntityStatus


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing."""
    def __init__(self, response_text: str = "", should_raise: Exception = None):
        self.response_text = response_text
        self.should_raise = should_raise

    @property
    def provider_name(self) -> str:
        return "mock_provider"

    @property
    def model_name(self) -> str:
        return "mock_model"

    def generate(self, prompt: str) -> str:
        if self.should_raise:
            raise self.should_raise
        return self.response_text

    def test_connection(self):
        return True, "Mock connected"


VALID_MOCK_REPORT = """
MEDICAL DOCUMENT SUMMARY

Symptoms:
- Fever

Medical Conditions:
- Not mentioned.

Medications:
- Not mentioned.

Dosages:
- Not mentioned.

Tests:
- Not mentioned.

Procedures:
- Not mentioned.

Body Parts:
- Not mentioned.

Important Findings:
- Not mentioned.

Brief Summary:
- The patient had fever.

Potential Missing Information:
- No medications or conditions were mentioned.

Safety Note:
- This report is generated from extracted document information and is not medical advice.
"""


class TestReportGenerator:
    """Test LLM report generation and fallback workflows."""

    def test_successful_llm_generation(self):
        client = MockLLMClient(response_text=VALID_MOCK_REPORT)
        generator = LLMReportGenerator(client=client)

        data = ExtractedMedicalData(
            symptoms=[MedicalEntity(text="fever", category="SYMPTOM", status=EntityStatus.PRESENT)]
        )

        report, is_fallback, err, val = generator.generate_report(data)
        assert is_fallback is False
        assert err is None
        assert "MEDICAL DOCUMENT SUMMARY" in report

    def test_automatic_fallback_on_api_timeout(self):
        client = MockLLMClient(should_raise=LLMTimeoutError("Request timed out"))
        generator = LLMReportGenerator(client=client)

        data = ExtractedMedicalData(
            symptoms=[MedicalEntity(text="fever", category="SYMPTOM", status=EntityStatus.PRESENT)]
        )

        report, is_fallback, err, val = generator.generate_report(data)
        assert is_fallback is True
        assert "timed out" in err.lower()
        assert "RULE-BASED FALLBACK REPORT" in report

    def test_automatic_fallback_on_invalid_output(self):
        # LLM returns malformed content missing required sections
        client = MockLLMClient(response_text="Here is a brief summary of the patient.")
        generator = LLMReportGenerator(client=client)

        data = ExtractedMedicalData(
            symptoms=[MedicalEntity(text="fever", category="SYMPTOM", status=EntityStatus.PRESENT)]
        )

        report, is_fallback, err, val = generator.generate_report(data)
        assert is_fallback is True
        assert "validation failed" in err.lower()
        assert "RULE-BASED FALLBACK REPORT" in report

    def test_zero_entities_triggers_fallback(self):
        client = MockLLMClient(response_text=VALID_MOCK_REPORT)
        generator = LLMReportGenerator(client=client)

        empty_data = ExtractedMedicalData()
        report, is_fallback, err, val = generator.generate_report(empty_data)
        assert is_fallback is True
        assert "RULE-BASED FALLBACK REPORT" in report
