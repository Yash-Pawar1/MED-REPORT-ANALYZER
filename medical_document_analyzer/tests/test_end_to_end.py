"""End-to-End Pipeline integration test from document ingestion to report validation."""

from pathlib import Path
import pytest

from src.pdf_processor import extract_document_text
from src.text_processor import preprocess_text
from src.entity_extractor import entity_extractor
from src.llm_report_generator import LLMReportGenerator
from src.report_validator import report_validator
from src.models import EntityStatus
from src.utils import get_project_root
from tests.test_report_generator import MockLLMClient


MOCK_E2E_LLM_REPORT = """
MEDICAL DOCUMENT SUMMARY

Symptoms:
- Fever
- Persistent cough
- Fatigue
- Chest pain (denied / not present)

Medical Conditions:
- Hypertension (historical)

Medications:
- Paracetamol

Dosages:
- 500 mg (for Paracetamol)

Tests:
- CBC
- Chest X-ray

Procedures:
- Not mentioned.

Body Parts:
- Chest

Important Findings:
- "Blood pressure was elevated during the examination."
- "Paracetamol 500 mg was prescribed for symptomatic relief."
- "A CBC and chest X-ray were recommended."

Brief Summary:
- The patient presented with fever, persistent cough, and fatigue.
- History of hypertension was documented and blood pressure was elevated.
- Paracetamol 500 mg was prescribed and CBC and chest X-ray were recommended.
- The patient denied chest pain.

Potential Missing Information:
- Surgical procedures and follow-up timeline were not mentioned.

Safety Note:
- This report is generated from extracted document information and is not medical advice.
"""


class TestEndToEndPipeline:
    """Full pipeline verification from source file to validated report."""

    def test_full_pipeline_with_sample_note(self):
        project_root = get_project_root()
        sample_path = project_root / "data" / "sample_medical_note.txt"
        assert sample_path.exists(), f"Sample note file missing at {sample_path}"

        # 1. Load sample document text
        with open(sample_path, "rb") as f:
            raw_bytes = f.read()
        extracted_text = extract_document_text("sample_medical_note.txt", raw_bytes)
        assert len(extracted_text) > 0

        # 2. Preprocess text
        cleaned_text, sentences = preprocess_text(extracted_text)
        assert len(sentences) >= 4

        # 3. Traditional NLP entity extraction (with negation, dosages, findings)
        structured_data = entity_extractor.extract(cleaned_text)

        # 4. Verify Symptoms (fever, persistent cough, fatigue as PRESENT)
        symptom_map = {s.text.lower(): s.status for s in structured_data.symptoms}
        assert "fever" in symptom_map
        assert symptom_map["fever"] == EntityStatus.PRESENT
        assert "persistent cough" in symptom_map
        assert symptom_map["persistent cough"] == EntityStatus.PRESENT
        assert "fatigue" in symptom_map
        assert symptom_map["fatigue"] == EntityStatus.PRESENT

        # 5. Verify Negated chest pain (status = NOT_PRESENT)
        assert "chest pain" in symptom_map
        assert symptom_map["chest pain"] == EntityStatus.NOT_PRESENT

        # 6. Verify Conditions (hypertension as HISTORICAL)
        cond_map = {c.text.lower(): c.status for c in structured_data.conditions}
        assert "hypertension" in cond_map
        assert cond_map["hypertension"] == EntityStatus.HISTORICAL

        # 7. Verify Medication
        med_names = [m.text.lower() for m in structured_data.medications]
        assert "paracetamol" in med_names

        # 8. Verify Dosage & Association
        assert len(structured_data.dosages) >= 1
        d = structured_data.dosages[0]
        assert d.value == "500 mg"
        assert d.medication is not None
        assert d.medication.lower() == "paracetamol"

        # 9. Verify Tests (CBC, chest X-ray)
        test_names = [t.text.lower() for t in structured_data.tests]
        assert "cbc" in test_names
        assert "chest x-ray" in test_names

        # 10. Verify Important Findings
        findings_sentences = [f.sentence for f in structured_data.important_findings]
        bp_found = any("elevated" in s.lower() for s in findings_sentences)
        assert bp_found is True

        # 11. Run LLM report generator with mock client
        mock_client = MockLLMClient(response_text=MOCK_E2E_LLM_REPORT)
        generator = LLMReportGenerator(client=mock_client)

        report, is_fallback, err_msg, validation = generator.generate_report(structured_data)

        # 12. Verify report validity and integrity
        assert is_fallback is False
        assert err_msg is None
        assert validation is not None
        assert validation.is_valid is True
        assert len(validation.missing_sections) == 0

        # Verify key sections in generated report
        assert "MEDICAL DOCUMENT SUMMARY" in report
        assert "Symptoms:" in report
        assert "Medical Conditions:" in report
        assert "Medications:" in report
        assert "Dosages:" in report
        assert "Tests:" in report
        assert "Important Findings:" in report
        assert "Brief Summary:" in report
        assert "Safety Note:" in report
