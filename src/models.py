"""Pydantic data models for structured medical NLP information and validation.

All extracted entities and findings are strictly validated before being
passed to formatting or LLM summarization.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class EntityStatus(str, Enum):
    """Status indicating presence, absence, or temporal nature of an entity."""
    PRESENT = "PRESENT"
    NOT_PRESENT = "NOT_PRESENT"
    HISTORICAL = "HISTORICAL"
    UNKNOWN = "UNKNOWN"


class EntityCategory(str, Enum):
    """Supported categories of extracted medical entities."""
    SYMPTOM = "SYMPTOM"
    CONDITION = "CONDITION"
    MEDICATION = "MEDICATION"
    DOSAGE = "DOSAGE"
    TEST = "TEST"
    PROCEDURE = "PROCEDURE"
    BODY_PART = "BODY_PART"
    IMPORTANT_FINDING = "IMPORTANT_FINDING"


class MedicalEntity(BaseModel):
    """Represents a validated medical entity with status and context."""
    text: str = Field(..., description="Text string of the extracted entity")
    category: str = Field(..., description="Entity category name (e.g. SYMPTOM, CONDITION)")
    status: EntityStatus = Field(default=EntityStatus.PRESENT, description="Presence status")
    sentence: str = Field(default="", description="Sentence context where entity appeared")
    start_char: Optional[int] = Field(default=None, description="Start character offset")
    end_char: Optional[int] = Field(default=None, description="End character offset")

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


class DosageInfo(BaseModel):
    """Structured dosage information associated with a medication."""
    value: str = Field(..., description="Dosage amount and unit (e.g. 500 mg, 10 ml)")
    frequency: Optional[str] = Field(default=None, description="Frequency (e.g. twice daily)")
    medication: Optional[str] = Field(default=None, description="Associated medication name")
    sentence: str = Field(default="", description="Sentence containing the dosage")

    @field_validator("value")
    @classmethod
    def strip_value(cls, v: str) -> str:
        return v.strip()


class MedicalFinding(BaseModel):
    """Important medical finding sentence with trigger indicator."""
    sentence: str = Field(..., description="Verbatim source sentence containing the finding")
    trigger_keyword: str = Field(..., description="Keyword triggering the finding extraction")

    @field_validator("sentence")
    @classmethod
    def strip_sentence(cls, v: str) -> str:
        return v.strip()


class ExtractedMedicalData(BaseModel):
    """Complete structured and validated medical data extracted via NLP."""
    symptoms: List[MedicalEntity] = Field(default_factory=list)
    conditions: List[MedicalEntity] = Field(default_factory=list)
    medications: List[MedicalEntity] = Field(default_factory=list)
    dosages: List[DosageInfo] = Field(default_factory=list)
    tests: List[MedicalEntity] = Field(default_factory=list)
    procedures: List[MedicalEntity] = Field(default_factory=list)
    body_parts: List[MedicalEntity] = Field(default_factory=list)
    important_findings: List[MedicalFinding] = Field(default_factory=list)
    raw_text_length: int = Field(default=0, description="Character count of cleaned source text")

    def total_entities_count(self) -> int:
        """Count total extracted entities across all categories."""
        return (
            len(self.symptoms)
            + len(self.conditions)
            + len(self.medications)
            + len(self.dosages)
            + len(self.tests)
            + len(self.procedures)
            + len(self.body_parts)
            + len(self.important_findings)
        )

    def to_llm_payload_dict(self) -> Dict[str, Any]:
        """Convert the extracted medical data to a clean dictionary for prompt injection.
        
        Strictly contains structured extracted facts, preserving negation and status.
        Never includes raw file content, paths, or system metadata.
        """
        def format_entity(e: MedicalEntity) -> Dict[str, str]:
            res = {"name": e.text, "status": e.status.value}
            return res

        return {
            "symptoms": [format_entity(e) for e in self.symptoms],
            "conditions": [format_entity(e) for e in self.conditions],
            "medications": [format_entity(e) for e in self.medications],
            "dosages": [
                {
                    "value": d.value,
                    "frequency": d.frequency or "unspecified",
                    "medication": d.medication or "unspecified"
                }
                for d in self.dosages
            ],
            "tests": [format_entity(e) for e in self.tests],
            "procedures": [format_entity(e) for e in self.procedures],
            "body_parts": [format_entity(e) for e in self.body_parts],
            "important_findings": [
                {"sentence": f.sentence, "indicator": f.trigger_keyword}
                for f in self.important_findings
            ]
        }

    def to_formatted_bullet_text(self) -> str:
        """Render structured data as formatted human-readable bullet points for prompt."""
        lines = []

        def render_section(title: str, entities: List[MedicalEntity]):
            lines.append(f"{title}:")
            if not entities:
                lines.append("  - Not mentioned.")
            else:
                for e in entities:
                    status_note = ""
                    if e.status == EntityStatus.NOT_PRESENT:
                        status_note = " [STATUS: NOT_PRESENT / DENIED]"
                    elif e.status == EntityStatus.HISTORICAL:
                        status_note = " [STATUS: HISTORICAL]"
                    elif e.status == EntityStatus.UNKNOWN:
                        status_note = " [STATUS: UNKNOWN]"
                    lines.append(f"  - {e.text}{status_note}")

        render_section("Symptoms", self.symptoms)
        render_section("Medical Conditions", self.conditions)
        render_section("Medications", self.medications)

        lines.append("Dosages:")
        if not self.dosages:
            lines.append("  - Not mentioned.")
        else:
            for d in self.dosages:
                desc = f"  - {d.value}"
                if d.medication:
                    desc += f" (for {d.medication})"
                if d.frequency:
                    desc += f", frequency: {d.frequency}"
                lines.append(desc)

        render_section("Tests", self.tests)
        render_section("Procedures", self.procedures)
        render_section("Body Parts", self.body_parts)

        lines.append("Important Findings:")
        if not self.important_findings:
            lines.append("  - Not mentioned.")
        else:
            for f in self.important_findings:
                lines.append(f'  - "{f.sentence}"')

        return "\n".join(lines)


class ReportValidationResult(BaseModel):
    """Validation report assessment results."""
    is_valid: bool = Field(..., description="Whether the report meets all criteria")
    missing_sections: List[str] = Field(default_factory=list)
    prohibited_terms_found: List[str] = Field(default_factory=list)
    hallucination_warnings: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
