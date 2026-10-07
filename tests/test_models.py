"""Tests for Pydantic models and structured data payload serialization."""

import pytest
from src.models import (
    MedicalEntity,
    DosageInfo,
    MedicalFinding,
    ExtractedMedicalData,
    EntityStatus,
    ReportValidationResult,
)


class TestModels:
    """Test validation and serialization methods on Pydantic models."""

    def test_medical_entity_validation(self):
        entity = MedicalEntity(
            text="  fever  ",
            category="SYMPTOM",
            status=EntityStatus.PRESENT,
            sentence="Patient has fever."
        )
        assert entity.text == "fever"
        assert entity.status == EntityStatus.PRESENT
        assert entity.category == "SYMPTOM"

    def test_dosage_info_validation(self):
        dosage = DosageInfo(
            value="  500 mg  ",
            frequency="twice daily",
            medication="Paracetamol",
            sentence="Paracetamol 500 mg twice daily."
        )
        assert dosage.value == "500 mg"
        assert dosage.frequency == "twice daily"
        assert dosage.medication == "Paracetamol"

    def test_extracted_medical_data_payload_serialization(self):
        data = ExtractedMedicalData(
            symptoms=[
                MedicalEntity(text="fever", category="SYMPTOM", status=EntityStatus.PRESENT),
                MedicalEntity(text="chest pain", category="SYMPTOM", status=EntityStatus.NOT_PRESENT),
            ],
            conditions=[
                MedicalEntity(text="hypertension", category="CONDITION", status=EntityStatus.HISTORICAL),
            ],
            medications=[
                MedicalEntity(text="Paracetamol", category="MEDICATION", status=EntityStatus.PRESENT),
            ],
            dosages=[
                DosageInfo(value="500 mg", frequency="twice daily", medication="Paracetamol")
            ],
            important_findings=[
                MedicalFinding(sentence="Blood pressure was elevated.", trigger_keyword="elevated")
            ]
        )

        payload = data.to_llm_payload_dict()
        assert len(payload["symptoms"]) == 2
        assert payload["symptoms"][0]["name"] == "fever"
        assert payload["symptoms"][0]["status"] == "PRESENT"
        assert payload["symptoms"][1]["status"] == "NOT_PRESENT"
        assert payload["conditions"][0]["status"] == "HISTORICAL"
        assert payload["dosages"][0]["value"] == "500 mg"

        bullet_text = data.to_formatted_bullet_text()
        assert "fever" in bullet_text
        assert "[STATUS: NOT_PRESENT / DENIED]" in bullet_text
        assert "[STATUS: HISTORICAL]" in bullet_text
        assert "500 mg (for Paracetamol)" in bullet_text

    def test_report_validation_result_model(self):
        val = ReportValidationResult(
            is_valid=True,
            missing_sections=[],
            warnings=[]
        )
        assert val.is_valid is True
