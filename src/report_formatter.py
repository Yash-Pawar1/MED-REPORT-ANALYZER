"""Report formatting utilities including the deterministic Rule-Based Fallback Report generator.

Ensures that even if external Cloud LLM APIs encounter timeouts, authentication
issues, or outages, a fully structured factual report is generated directly
from verified NLP entities and clearly labeled.
"""

from datetime import datetime
from typing import Optional
from src.models import ExtractedMedicalData, EntityStatus
from src.utils import setup_logger

logger = setup_logger(__name__)


def generate_rule_based_fallback_report(
    data: ExtractedMedicalData,
    failure_reason: Optional[str] = None
) -> str:
    """Generate a 100% deterministic, non-LLM medical report from structured NLP data.
    
    Clearly labeled as 'Rule-Based Fallback Report'.
    Adheres strictly to the required section headers.
    """
    lines = [
        "MEDICAL DOCUMENT SUMMARY",
        "=== RULE-BASED FALLBACK REPORT (NO AI GENERATION) ===",
        "",
        "Symptoms:"
    ]

    if not data.symptoms:
        lines.append("- Not mentioned.")
    else:
        for s in data.symptoms:
            status_text = ""
            if s.status == EntityStatus.NOT_PRESENT:
                status_text = " (denied / not present)"
            elif s.status == EntityStatus.HISTORICAL:
                status_text = " (historical)"
            lines.append(f"- {s.text.capitalize()}{status_text}")

    lines.append("\nMedical Conditions:")
    if not data.conditions:
        lines.append("- Not mentioned.")
    else:
        for c in data.conditions:
            status_text = ""
            if c.status == EntityStatus.NOT_PRESENT:
                status_text = " (denied / not present)"
            elif c.status == EntityStatus.HISTORICAL:
                status_text = " (historical)"
            lines.append(f"- {c.text.capitalize()}{status_text}")

    lines.append("\nMedications:")
    if not data.medications:
        lines.append("- Not mentioned.")
    else:
        for m in data.medications:
            lines.append(f"- {m.text.capitalize()}")

    lines.append("\nDosages:")
    if not data.dosages:
        lines.append("- Not mentioned.")
    else:
        for d in data.dosages:
            entry = f"- {d.value}"
            if d.medication:
                entry += f" (associated with {d.medication})"
            if d.frequency:
                entry += f", frequency: {d.frequency}"
            lines.append(entry)

    lines.append("\nTests:")
    if not data.tests:
        lines.append("- Not mentioned.")
    else:
        for t in data.tests:
            lines.append(f"- {t.text}")

    lines.append("\nProcedures:")
    if not data.procedures:
        lines.append("- Not mentioned.")
    else:
        for p in data.procedures:
            lines.append(f"- {p.text.capitalize()}")

    lines.append("\nBody Parts:")
    if not data.body_parts:
        lines.append("- Not mentioned.")
    else:
        for b in data.body_parts:
            lines.append(f"- {b.text.capitalize()}")

    lines.append("\nImportant Findings:")
    if not data.important_findings:
        lines.append("- Not mentioned.")
    else:
        for f in data.important_findings:
            lines.append(f'- "{f.sentence}"')

    # Brief factual summary generated deterministically
    summary_sentences = []
    present_symptoms = [s.text for s in data.symptoms if s.status == EntityStatus.PRESENT]
    if present_symptoms:
        summary_sentences.append(f"Document mentions presenting symptom(s): {', '.join(present_symptoms)}.")

    hist_conditions = [c.text for c in data.conditions if c.status == EntityStatus.HISTORICAL]
    if hist_conditions:
        summary_sentences.append(f"Document notes historical condition(s): {', '.join(hist_conditions)}.")

    prescribed_meds = [m.text for m in data.medications]
    if prescribed_meds:
        summary_sentences.append(f"Document records medication(s): {', '.join(prescribed_meds)}.")

    negated_items = [
        item.text for item in (data.symptoms + data.conditions)
        if item.status == EntityStatus.NOT_PRESENT
    ]
    if negated_items:
        summary_sentences.append(f"Explicitly denied or absent: {', '.join(negated_items)}.")

    if not summary_sentences:
        summary_sentences.append("Deterministic extraction completed from document text.")

    lines.append("\nBrief Summary:")
    for sent in summary_sentences[:4]:
        lines.append(f"- {sent}")

    # Missing information detection
    missing = []
    if not data.symptoms:
        missing.append("symptoms")
    if not data.conditions:
        missing.append("medical conditions")
    if not data.medications:
        missing.append("medications")
    if not data.dosages:
        missing.append("dosages")
    if not data.tests:
        missing.append("tests/lab work")
    if not data.procedures:
        missing.append("procedures")

    lines.append("\nPotential Missing Information:")
    if missing:
        lines.append(f"- No explicit mention of {', '.join(missing)} was detected.")
    else:
        lines.append("- All standard entity categories had one or more extracted items.")

    lines.append("\nSafety Note:")
    lines.append("- This report is generated from extracted document information and is not medical advice.")

    if failure_reason:
        lines.append(f"\n[Generation Notice: Generated via Rule-Based NLP engine due to: {failure_reason}]")

    return "\n".join(lines)


def format_export_download_text(
    report_text: str,
    filename: str,
    provider_name: str,
    model_name: str,
    is_fallback: bool
) -> str:
    """Format full export text for file download including metadata header and safety notice."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        "=" * 70,
        "MEDICAL DOCUMENT ANALYSIS REPORT",
        "=" * 70,
        f"Source Document: {filename}",
        f"Generated At:    {timestamp}",
        f"Generation Mode: {'Rule-Based Fallback (No AI)' if is_fallback else f'Cloud LLM ({provider_name.upper()} - {model_name})'}",
        "=" * 70,
        "",
        report_text,
        "",
        "=" * 70,
        "DISCLAIMER: This report is an automated extraction summary intended",
        "for academic demonstration. It does not provide medical diagnoses,",
        "clinical decision support, or treatment recommendations.",
        "=" * 70,
    ]
    return "\n".join(header)
