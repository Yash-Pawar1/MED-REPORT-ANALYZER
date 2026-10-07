"""Tests for regex-based dosage extraction, frequency detection, and medication association."""

import pytest
from src.dosage_extractor import DosageExtractor


@pytest.fixture
def extractor():
    return DosageExtractor()


class TestDosageExtractor:
    """Test dosage regex accuracy and intra-sentence medication association."""

    def test_extract_milligrams(self, extractor):
        sentence = "Prescribed 500 mg for relief."
        dosages = extractor.extract_from_sentence(sentence)
        assert len(dosages) == 1
        assert dosages[0].value == "500 mg"

    def test_extract_milliliters(self, extractor):
        sentence = "Administer 10 ml orally."
        dosages = extractor.extract_from_sentence(sentence)
        assert len(dosages) == 1
        assert dosages[0].value == "10 ml"

    def test_extract_tablets(self, extractor):
        sentence = "Take 2 tablets before sleep."
        dosages = extractor.extract_from_sentence(sentence)
        assert len(dosages) == 1
        assert dosages[0].value == "2 tablets"

    def test_extract_dosage_and_frequency(self, extractor):
        sentence = "Patient takes 20 mg/day twice daily."
        dosages = extractor.extract_from_sentence(sentence)
        assert len(dosages) == 1
        assert dosages[0].value == "20 mg/day"
        assert dosages[0].frequency == "twice daily"

    def test_medication_association_in_same_sentence(self, extractor):
        sentence = "Patient takes Paracetamol 500 mg twice daily."
        dosages = extractor.extract_from_sentence(sentence, known_medications=["Paracetamol"])
        assert len(dosages) == 1
        assert dosages[0].value == "500 mg"
        assert dosages[0].frequency == "twice daily"
        assert dosages[0].medication == "Paracetamol"

    def test_reject_arbitrary_numbers(self, extractor):
        """Numbers without clinical units must NOT be extracted as dosages."""
        sentence = "Patient is 45 years old, staying in room 204 for 3 days."
        dosages = extractor.extract_from_sentence(sentence)
        assert len(dosages) == 0

    def test_no_cross_sentence_association(self, extractor):
        """Medication in sentence 1 should not falsely attach to dosage in sentence 2."""
        sents = [
            "Patient was diagnosed and prescribed Paracetamol.",
            "Later, 10 ml of oral solution was administered."
        ]
        dosages = extractor.extract_dosages(sents, known_medications=["Paracetamol"])
        # The 10 ml in sentence 2 should NOT be associated with Paracetamol from sentence 1
        saline_dosage = [d for d in dosages if d.value == "10 ml"][0]
        assert saline_dosage.medication is None
