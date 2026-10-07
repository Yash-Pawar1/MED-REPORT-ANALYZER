"""Tests for report output validation, structural requirements, and hallucination checks."""

import pytest
from src.report_validator import ReportValidator
from src.models import ExtractedMedicalData, MedicalEntity, EntityStatus


@pytest.fixture
def validator():
    return ReportValidator()


@pytest.fixture
def sample_extracted_data():
    return ExtractedMedicalData(
        symptoms=[MedicalEntity(text="fever", category="SYMPTOM", status=EntityStatus.PRESENT)],
        conditions=[MedicalEntity(text="hypertension", category="CONDITION", status=EntityStatus.HISTORICAL)],
        medications=[MedicalEntity(text="Paracetamol", category="MEDICATION", status=EntityStatus.PRESENT)]
    )


VALID_REPORT_TEXT = """
MEDICAL DOCUMENT SUMMARY

Symptoms:
- Fever

Medical Conditions:
- Hypertension (historical)

Medications:
- Paracetamol

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
- The patient had fever and a history of hypertension. Paracetamol was recorded.

Potential Missing Information:
- Diagnostic tests and procedures were not mentioned.

Safety Note:
- This report is generated from extracted document information and is not medical advice.
"""


class TestReportValidator:
    """Test output validation logic and defensive checks."""

    def test_valid_report_passes(self, validator, sample_extracted_data):
        result = validator.validate(VALID_REPORT_TEXT, sample_extracted_data)
        assert result.is_valid is True
        assert len(result.missing_sections) == 0
        assert len(result.prohibited_terms_found) == 0

    def test_empty_report_fails(self, validator, sample_extracted_data):
        result = validator.validate("", sample_extracted_data)
        assert result.is_valid is False
        assert len(result.warnings) > 0

    def test_missing_required_section_fails(self, validator, sample_extracted_data):
        # Remove Safety Note
        bad_report = VALID_REPORT_TEXT.replace("Safety Note:", "Disclaimer:")
        result = validator.validate(bad_report, sample_extracted_data)
        assert result.is_valid is False
        assert "Safety Note:" in result.missing_sections

    def test_prohibited_diagnostic_language_fails(self, validator, sample_extracted_data):
        bad_report = VALID_REPORT_TEXT + "\nPatient likely has pneumonia and needs treatment recommendation."
        result = validator.validate(bad_report, sample_extracted_data)
        assert result.is_valid is False
        assert len(result.prohibited_terms_found) > 0

    def test_hallucination_warning_on_unmentioned_category(self, validator):
        # Data with zero symptoms
        empty_symp_data = ExtractedMedicalData()
        # Report lists a symptom
        report = VALID_REPORT_TEXT.replace("- Not mentioned.", "- Cough")
        result = validator.validate(report, empty_symp_data)
        # Should flag hallucination warning for symptom
        assert len(result.hallucination_warnings) > 0
