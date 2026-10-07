"""Extraction of important medical finding sentences.

Identifies key clinical observations, lab trends, and physical examination findings
by checking for clinical indicator keywords while strictly preserving the verbatim
source sentence without rewriting into diagnoses.
"""

import re
from typing import List, Set
from src.models import MedicalFinding
from src.config import config
from src.utils import setup_logger

logger = setup_logger(__name__)

DEFAULT_FINDING_KEYWORDS = [
    "diagnosed",
    "positive",
    "negative",
    "elevated",
    "reduced",
    "abnormal",
    "normal",
    "detected",
    "prescribed",
    "recommended",
    "history of",
    "suffering from",
    "increased",
    "decreased",
    "high",
    "low"
]


class FindingsExtractor:
    """Extracts factual finding sentences containing clinical indicators."""

    def __init__(self, keywords: List[str] = None):
        self.keywords = keywords or config.finding_keywords or DEFAULT_FINDING_KEYWORDS
        # Compile word boundary regexes for accurate matching
        self.keyword_patterns = [
            (kw, re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE))
            for kw in self.keywords
        ]

    def extract_findings(self, sentences: List[str]) -> List[MedicalFinding]:
        """Extract sentences that contain one or more finding indicator keywords.
        
        Preserves original sentence text exactly without modification or diagnosis inference.
        """
        findings: List[MedicalFinding] = []
        seen_sentences: Set[str] = set()

        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue

            normalized_key = sent_clean.lower()
            if normalized_key in seen_sentences:
                continue

            # Check if any indicator keyword matches
            matched_trigger = None
            for kw, pattern in self.keyword_patterns:
                if pattern.search(sent_clean):
                    matched_trigger = kw
                    break

            if matched_trigger:
                seen_sentences.add(normalized_key)
                findings.append(
                    MedicalFinding(
                        sentence=sent_clean,
                        trigger_keyword=matched_trigger
                    )
                )

        return findings


# Default singleton instance
findings_extractor = FindingsExtractor()
