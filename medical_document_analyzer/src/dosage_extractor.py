"""Dosage and frequency extraction using robust regular expressions.

Extracts medical dosages, administration frequencies, and associates them
strictly with medications located within the same local sentence context.
Arbitrary numbers without clinical units are strictly excluded.
"""

import re
from typing import List, Optional, Set
from src.models import DosageInfo
from src.utils import setup_logger

logger = setup_logger(__name__)

# Explicit regex for valid medical dosage amounts with units
DOSAGE_VALUE_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?\s*(?:mg(?:\/(?:day|kg))?|g|mcg|micrograms?|ml|mL|tablets?|capsules?|pills?|drops?|puffs?|units?|IU))\b",
    re.IGNORECASE
)

# Frequency indicators
FREQUENCY_PATTERNS = [
    r"\btwice\s+daily\b",
    r"\bonce\s+daily\b",
    r"\bdaily\b",
    r"\bevery\s+\d+\s*(?:hours|hrs|h)\b",
    r"\bthree\s+times\s+(?:a\s+)?day\b",
    r"\bfour\s+times\s+(?:a\s+)?day\b",
    r"\bonce\s+(?:a\s+)?day\b",
    r"\btwice\s+(?:a\s+)?day\b",
    r"\bbefore\s+meals\b",
    r"\bafter\s+meals\b",
    r"\bat\s+bedtime\b",
    r"\bas\s+needed\b",
    r"\bq\d+h\b",
    r"\bbid\b",
    r"\btid\b",
    r"\bqid\b",
    r"\bprn\b",
]

COMBINED_FREQUENCY_REGEX = re.compile(
    "|".join(f"({p})" for p in FREQUENCY_PATTERNS),
    re.IGNORECASE
)


class DosageExtractor:
    """Extracts dosage values, frequencies, and associates them with local medications."""

    def __init__(self):
        self.dosage_regex = DOSAGE_VALUE_PATTERN
        self.frequency_regex = COMBINED_FREQUENCY_REGEX

    def extract_from_sentence(
        self,
        sentence: str,
        known_medications: Optional[List[str]] = None
    ) -> List[DosageInfo]:
        """Extract all dosages from a single sentence and correlate with local medications."""
        if not sentence or not sentence.strip():
            return []

        dosages: List[DosageInfo] = []
        known_meds = known_medications or []

        # Find all frequency matches in this sentence
        freq_matches = list(self.frequency_regex.finditer(sentence))

        # Find all dosage value matches
        for d_match in self.dosage_regex.finditer(sentence):
            dosage_val = d_match.group(1).strip()
            d_start = d_match.start()
            d_end = d_match.end()

            # 1. Associate nearest frequency in the sentence
            matched_freq: Optional[str] = None
            if freq_matches:
                # Find closest frequency match to dosage position
                closest_freq = min(
                    freq_matches,
                    key=lambda m: min(abs(m.start() - d_end), abs(m.end() - d_start))
                )
                matched_freq = closest_freq.group(0).strip()

            # 2. Associate nearest medication in the same sentence
            matched_med: Optional[str] = None
            if known_meds:
                best_distance = float("inf")
                for med in known_meds:
                    for m_match in re.finditer(r"\b" + re.escape(med) + r"\b", sentence, re.IGNORECASE):
                        # Measure distance between medication and dosage
                        dist = min(abs(m_match.start() - d_end), abs(m_match.end() - d_start))
                        if dist < best_distance:
                            best_distance = dist
                            matched_med = med

            dosages.append(
                DosageInfo(
                    value=dosage_val,
                    frequency=matched_freq,
                    medication=matched_med,
                    sentence=sentence.strip()
                )
            )

        return dosages

    def extract_dosages(
        self,
        sentences: List[str],
        known_medications: Optional[List[str]] = None
    ) -> List[DosageInfo]:
        """Extract dosages across all document sentences, preventing cross-sentence bleeding."""
        all_dosages: List[DosageInfo] = []
        seen_values: Set[Tuple[str, Optional[str], Optional[str]]] = set()

        for sent in sentences:
            sent_dosages = self.extract_from_sentence(sent, known_medications=known_medications)
            for d in sent_dosages:
                key = (d.value.lower(), (d.frequency or "").lower(), (d.medication or "").lower())
                if key not in seen_values:
                    seen_values.add(key)
                    all_dosages.append(d)

        return all_dosages


# Default singleton instance
dosage_extractor = DosageExtractor()
