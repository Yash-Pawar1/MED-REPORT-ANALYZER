"""Tests for negation and historical temporal status detection."""

import pytest
from src.negation import NegationDetector
from src.models import EntityStatus


@pytest.fixture
def detector():
    return NegationDetector()


class TestNegation:
    """Test rule-based negation, history, and affirmative statuses."""

    def test_affirmative_status(self, detector):
        sentence = "The patient presented with fever and cough."
        status = detector.detect_status("fever", sentence)
        assert status == EntityStatus.PRESENT

    def test_pre_negation_no(self, detector):
        sentence = "The patient has no fever."
        status = detector.detect_status("fever", sentence)
        assert status == EntityStatus.NOT_PRESENT

    def test_denies_symptom(self, detector):
        sentence = "The patient denies chest pain."
        status = detector.detect_status("chest pain", sentence)
        assert status == EntityStatus.NOT_PRESENT

    def test_no_evidence_of(self, detector):
        sentence = "No evidence of pneumonia on examination."
        status = detector.detect_status("pneumonia", sentence)
        assert status == EntityStatus.NOT_PRESENT

    def test_post_negation_ruled_out(self, detector):
        sentence = "Pneumonia was ruled out by the team."
        status = detector.detect_status("pneumonia", sentence)
        assert status == EntityStatus.NOT_PRESENT

    def test_historical_status_history_of(self, detector):
        sentence = "The patient has a history of hypertension."
        status = detector.detect_status("hypertension", sentence)
        assert status == EntityStatus.HISTORICAL

    def test_historical_status_previously_had(self, detector):
        sentence = "The patient previously had asthma as a child."
        status = detector.detect_status("asthma", sentence)
        assert status == EntityStatus.HISTORICAL

    def test_clause_boundary_preserves_affirmative(self, detector):
        sentence = "The patient denies headache, but reports severe fever."
        # Headache is denied
        headache_status = detector.detect_status("headache", sentence)
        assert headache_status == EntityStatus.NOT_PRESENT
        # Fever is reported, separated by clause boundary 'but'
        fever_status = detector.detect_status("fever", sentence)
        assert fever_status == EntityStatus.PRESENT
