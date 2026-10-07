"""Tests for text normalization, sentence segmentation, and negation preservation."""

import pytest
from src.text_processor import (
    normalize_unicode,
    normalize_whitespace,
    segment_sentences,
    tokenize_sentence,
    preprocess_text,
    PRESERVED_NEGATION_WORDS,
)


class TestTextProcessor:
    """Test text processing and normalization functions."""

    def test_normalize_unicode(self):
        # Test non-breaking spaces and unicode characters
        text = "Blood\u00a0pressure was 120/80\u202fmmHg."
        cleaned = normalize_unicode(text)
        assert "\u00a0" not in cleaned
        assert "\u202f" not in cleaned
        assert "Blood pressure was 120/80 mmHg." == cleaned

    def test_normalize_whitespace(self):
        text = "Patient has    fever.\n\n\n\nNo   cough.\r\nNext   line."
        cleaned = normalize_whitespace(text)
        assert "  " not in cleaned
        assert "\r" not in cleaned
        assert "\n\n\n" not in cleaned
        assert "Patient has fever." in cleaned

    def test_segment_sentences(self):
        text = "Patient presented with fever. The patient has a history of hypertension. Denies chest pain."
        sentences = segment_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Patient presented with fever."
        assert sentences[1] == "The patient has a history of hypertension."
        assert sentences[2] == "Denies chest pain."

    def test_preservation_of_negation_words(self):
        """Verify that negation words are strictly kept intact."""
        raw = "The patient has no fever, not coughing, and denies shortness of breath without fatigue."
        cleaned, sents = preprocess_text(raw)
        words = [w.lower() for w in tokenize_sentence(cleaned)]
        for neg in ["no", "not", "denies", "without"]:
            assert neg in words
            assert neg in PRESERVED_NEGATION_WORDS

    def test_preservation_of_medical_tokens(self):
        sentence = "Prescribed 500 mg/day for 5 days."
        tokens = tokenize_sentence(sentence)
        assert "mg/day" in tokens or "mg" in tokens

    def test_empty_text_preprocessing(self):
        cleaned, sents = preprocess_text("")
        assert cleaned == ""
        assert sents == []

    def test_whitespace_only_preprocessing(self):
        cleaned, sents = preprocess_text("   \n\t   ")
        assert cleaned == ""
        assert sents == []
