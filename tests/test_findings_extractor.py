"""Tests for medical finding sentence extraction and verbatim preservation."""

import pytest
from src.findings_extractor import FindingsExtractor


@pytest.fixture
def extractor():
    return FindingsExtractor()


class TestFindingsExtractor:
    """Test finding indicator extraction and source preservation."""

    def test_extract_elevated_finding(self, extractor):
        sentences = [
            "The patient arrived at 9am.",
            "Blood pressure was elevated during the examination."
        ]
        findings = extractor.extract_findings(sentences)
        assert len(findings) == 1
        assert findings[0].sentence == "Blood pressure was elevated during the examination."
        assert findings[0].trigger_keyword == "elevated"

    def test_extract_prescribed_and_recommended(self, extractor):
        sentences = [
            "Paracetamol 500 mg was prescribed for symptomatic relief.",
            "A CBC and chest X-ray were recommended."
        ]
        findings = extractor.extract_findings(sentences)
        assert len(findings) == 2
        triggers = [f.trigger_keyword for f in findings]
        assert "prescribed" in triggers
        assert "recommended" in triggers

    def test_verbatim_sentence_preservation(self, extractor):
        raw_sentence = "Lab results showed abnormal liver enzymes."
        findings = extractor.extract_findings([raw_sentence])
        assert len(findings) == 1
        assert findings[0].sentence == raw_sentence

    def test_ignore_sentences_without_indicators(self, extractor):
        sentences = [
            "The patient was accompanied by family.",
            "Follow-up is scheduled for next Tuesday."
        ]
        findings = extractor.extract_findings(sentences)
        assert len(findings) == 0
