"""Output validation and defensive hallucination checks for generated medical reports.

Validates that all mandatory sections are present, prevents unauthorized
diagnostic/prescriptive language, and flags suspicious entities not present
in the structured input data.
"""

import re
from typing import List, Set
from src.models import ExtractedMedicalData, ReportValidationResult
from src.config import config
from src.utils import setup_logger

logger = setup_logger(__name__)

# Phrases that imply autonomous medical diagnosis or prescription
PROHIBITED_PHRASES = [
    r"\bpatient\s+(?:likely|probably)\s+has\b",
    r"\bsuspected\s+diagnosis\s+of\b",
    r"\brecommend\s+(?:starting|prescribing|taking)\b",
    r"\bsuggest\s+(?:increasing|decreasing)\s+(?:the\s+)?dose\b",
    r"\btreatment\s+recommendation\b",
    r"\bprescribe\s+(?:additional|new)\b",
    r"\bshould\s+take\b",
    r"\bmust\s+take\b",
    r"\bprognosis\s+is\b",
    r"\brisk\s+of\s+death\b",
    r"\bdrug\s+interaction\s+detected\b",
]


class ReportValidator:
    """Validates generated report structure, safety constraints, and factual consistency."""

    def __init__(self, required_sections: List[str] = None):
        self.required_sections = required_sections or config.required_report_sections

    def validate(self, report_text: str, structured_data: ExtractedMedicalData) -> ReportValidationResult:
        """Validate LLM-generated report text against structure and safety constraints.
        
        Args:
            report_text: Raw output string from LLM
            structured_data: The source ExtractedMedicalData object supplied to the prompt
            
        Returns:
            ReportValidationResult detailing validity, missing sections, and warnings
        """
        if not report_text or not report_text.strip():
            return ReportValidationResult(
                is_valid=False,
                missing_sections=self.required_sections,
                warnings=["The generated report is empty."]
            )

        missing_sections: List[str] = []
        # 1. Section presence validation
        for section in self.required_sections:
            if section.lower() not in report_text.lower():
                missing_sections.append(section)

        # 2. Prohibited diagnostic/prescriptive language check
        prohibited_found: List[str] = []
        for phrase_pattern in PROHIBITED_PHRASES:
            match = re.search(phrase_pattern, report_text, re.IGNORECASE)
            if match:
                prohibited_found.append(match.group(0))

        # 3. Defensive entity hallucination check
        hallucination_warnings: List[str] = []
        warnings: List[str] = []

        # Build a known whitelist of extracted entity strings (lower-case)
        known_entity_strings: Set[str] = set()
        for cat_list in [
            structured_data.symptoms,
            structured_data.conditions,
            structured_data.medications,
            structured_data.tests,
            structured_data.procedures,
            structured_data.body_parts,
        ]:
            for item in cat_list:
                known_entity_strings.add(item.text.lower())
                # Add single words for multi-word phrases
                for word in item.text.lower().split():
                    if len(word) > 3:
                        known_entity_strings.add(word)

        for d in structured_data.dosages:
            known_entity_strings.add(d.value.lower())
            if d.medication:
                known_entity_strings.add(d.medication.lower())

        # Check section bullet points for obvious hallucinated terms
        lines = report_text.splitlines()
        current_section = ""
        for line in lines:
            line_str = line.strip()
            # Track header
            for sec in ["Symptoms:", "Medical Conditions:", "Medications:", "Tests:", "Procedures:"]:
                if sec.lower() in line_str.lower():
                    current_section = sec
                    break

            if current_section and line_str.startswith("-"):
                bullet_content = line_str.lstrip("-").strip()
                if "not mentioned" in bullet_content.lower():
                    continue
                # If bullet content mentions an entity when that whole category in structured data was empty:
                if current_section == "Symptoms:" and not structured_data.symptoms:
                    hallucination_warnings.append(f"Symptoms section lists '{bullet_content}' but no symptoms were in input.")
                elif current_section == "Medical Conditions:" and not structured_data.conditions:
                    hallucination_warnings.append(f"Conditions section lists '{bullet_content}' but no conditions were in input.")
                elif current_section == "Medications:" and not structured_data.medications:
                    hallucination_warnings.append(f"Medications section lists '{bullet_content}' but no medications were in input.")
                elif current_section == "Tests:" and not structured_data.tests:
                    hallucination_warnings.append(f"Tests section lists '{bullet_content}' but no tests were in input.")
                elif current_section == "Procedures:" and not structured_data.procedures:
                    hallucination_warnings.append(f"Procedures section lists '{bullet_content}' but no procedures were in input.")

        # Determine validity:
        # A report is invalid if missing essential sections or using prohibited diagnosis phrases
        is_valid = (len(missing_sections) == 0) and (len(prohibited_found) == 0)

        if not is_valid:
            if missing_sections:
                warnings.append(f"Missing required sections: {', '.join(missing_sections)}")
            if prohibited_found:
                warnings.append(f"Prohibited prescriptive/diagnostic phrasing detected: {', '.join(prohibited_found)}")

        return ReportValidationResult(
            is_valid=is_valid,
            missing_sections=missing_sections,
            prohibited_terms_found=prohibited_found,
            hallucination_warnings=hallucination_warnings,
            warnings=warnings
        )


# Default singleton instance
report_validator = ReportValidator()
