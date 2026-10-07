"""Text preprocessing and normalization pipeline.

Cleans extracted text, standardizes Unicode, normalizes whitespace,
and segments text into sentences while strictly preserving negation terms
and medical punctuation.
"""

import re
import unicodedata
from typing import List, Tuple
import nltk

from src.utils import setup_logger

logger = setup_logger(__name__)

# Ensure NLTK punkt tokenizer is available safely without crash
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception as e:
        logger.warning(f"Could not download NLTK punkt data: {e}. Fallback regex will be used.")

# Medical negation keywords that MUST NEVER be removed or altered
PRESERVED_NEGATION_WORDS = {
    "no", "not", "without", "denies", "denied", "negative", "absent",
    "never", "none", "neither", "nor", "rules", "ruled"
}


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to standard NFKC representation."""
    if not text:
        return ""
    # Standardize curly quotes, dashes, non-breaking spaces
    normalized = unicodedata.normalize("NFKC", text)
    # Replace non-breaking spaces with standard space
    normalized = normalized.replace("\u00a0", " ").replace("\u202f", " ")
    return normalized


def normalize_whitespace(text: str) -> str:
    """Normalize line endings, tabs, and excess spacing while preserving paragraphs."""
    if not text:
        return ""
    # Convert carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace tabs and multiple spaces with a single space per line
    lines = []
    for line in text.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)
    
    # Reassemble preserving paragraph breaks (empty lines)
    result = "\n".join(lines)
    # Collapse 3+ newlines to 2 newlines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def segment_sentences(text: str) -> List[str]:
    """Segment text into individual sentences with abbreviation handling.
    
    Uses NLTK sent_tokenize with robust fallback to regex.
    """
    if not text or not text.strip():
        return []

    cleaned_text = normalize_whitespace(normalize_unicode(text))
    # Replace single line breaks that occur mid-sentence with space
    flattened_text = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned_text)

    sentences = []
    try:
        raw_sents = nltk.sent_tokenize(flattened_text)
        sentences = [s.strip() for s in raw_sents if s.strip()]
    except Exception:
        # Fallback regex sentence splitter protecting medical abbreviations like 'mg.' or 'Dr.'
        pattern = r"(?<=[.!?])\s+(?=[A-Z0-9])"
        raw_sents = re.split(pattern, flattened_text)
        sentences = [s.strip() for s in raw_sents if s.strip()]

    # Further clean each sentence
    cleaned_sents = []
    for s in sentences:
        s_clean = re.sub(r"\s+", " ", s).strip()
        if len(s_clean) > 0:
            cleaned_sents.append(s_clean)

    return cleaned_sents


def tokenize_sentence(sentence: str) -> List[str]:
    """Tokenize a sentence into words preserving medical tokens like 'mg/day' and hyphenated words."""
    if not sentence:
        return []
    # Match words, numbers with decimals, hyphenated terms, and slashed terms (mg/day)
    token_pattern = r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*|[.,!?;]"
    return re.findall(token_pattern, sentence)


def preprocess_text(raw_text: str) -> Tuple[str, List[str]]:
    """Complete preprocessing pipeline for input text.
    
    Returns:
        Tuple of (clean_normalized_text, list_of_sentences)
    """
    if not raw_text:
        return "", []

    normalized = normalize_unicode(raw_text)
    cleaned = normalize_whitespace(normalized)
    sentences = segment_sentences(cleaned)
    return cleaned, sentences
