"""Tests for traditional spaCy EntityRuler medical entity extraction."""

import pytest
from src.entity_extractor import MedicalEntityExtractor
from src.models import EntityStatus


@pytest.fixture
def extractor():
    return MedicalEntityExtractor()


class TestEntityExtractor:
    """Test entity extraction across clinical categories."""

    def test_extract_symptoms(self, extractor):
        text = "Patient presented with fever, persistent cough, and severe fatigue."
        data = extractor.extract(text)
        symptom_texts = [s.text.lower() for s in data.symptoms]
        assert "fever" in symptom_texts
        assert "persistent cough" in symptom_texts
        assert "fatigue" in symptom_texts

    def test_extract_conditions(self, extractor):
        text = "Patient has hypertension and diabetes."
        data = extractor.extract(text)
        cond_texts = [c.text.lower() for c in data.conditions]
        assert "hypertension" in cond_texts
        assert "diabetes" in cond_texts

    def test_extract_medications(self, extractor):
        text = "Administered Paracetamol and started Metformin."
        data = extractor.extract(text)
        med_texts = [m.text.lower() for m in data.medications]
        assert "paracetamol" in med_texts
        assert "metformin" in med_texts

    def test_extract_tests(self, extractor):
        text = "Ordered a CBC and a chest X-ray for diagnostic evaluation."
        data = extractor.extract(text)
        test_texts = [t.text.lower() for t in data.tests]
        assert "cbc" in test_texts
        assert "chest x-ray" in test_texts

    def test_extract_procedures(self, extractor):
        text = "Patient scheduled for endoscopy and skin biopsy."
        data = extractor.extract(text)
        proc_texts = [p.text.lower() for p in data.procedures]
        assert "endoscopy" in proc_texts
        assert "biopsy" in proc_texts

    def test_extract_body_parts(self, extractor):
        text = "Examination of the chest, lungs, and heart was performed."
        data = extractor.extract(text)
        body_texts = [b.text.lower() for b in data.body_parts]
        assert "chest" in body_texts
        assert "lungs" in body_texts
        assert "heart" in body_texts

    def test_case_insensitive_deduplication(self, extractor):
        text = "Fever noted in morning. Later, fever worsened. High FEVER confirmed."
        data = extractor.extract(text)
        assert len(data.symptoms) == 1
        assert data.symptoms[0].text in ["Fever", "fever", "FEVER"]

    def test_empty_text_extraction(self, extractor):
        data = extractor.extract("")
        assert len(data.symptoms) == 0
        assert len(data.conditions) == 0
        assert data.total_entities_count() == 0
