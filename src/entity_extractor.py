"""Traditional NLP medical entity extraction using spaCy EntityRuler and rules.

Extracts symptoms, conditions, medications, tests, procedures, and body parts
without using black-box LLMs or deep learning NER training. All entities are mapped
to validated Pydantic models with status detection.
"""

from typing import Dict, List, Optional, Set, Tuple
import spacy
from spacy.pipeline import EntityRuler

from src.models import (
    EntityCategory,
    EntityStatus,
    ExtractedMedicalData,
    MedicalEntity,
)
from src.config import config
from src.negation import negation_detector
from src.dosage_extractor import dosage_extractor
from src.findings_extractor import findings_extractor
from src.text_processor import preprocess_text
from src.utils import setup_logger

logger = setup_logger(__name__)

# Default comprehensive vocabulary fallback
DEFAULT_VOCABULARY = {
    "SYMPTOM": [
        "fever", "headache", "cough", "persistent cough", "chronic cough",
        "fatigue", "nausea", "vomiting", "dizziness", "chest pain",
        "abdominal pain", "shortness of breath", "dyspnea", "sore throat",
        "weakness", "chills", "sweats", "rash", "diarrhea", "constipation",
        "back pain", "joint pain", "wheezing", "palpitations", "swelling"
    ],
    "CONDITION": [
        "hypertension", "diabetes", "diabetes mellitus", "asthma", "pneumonia",
        "anemia", "migraine", "arthritis", "infection", "influenza",
        "bronchitis", "copd", "tuberculosis", "hypothyroidism",
        "hyperthyroidism", "coronary artery disease", "heart failure",
        "arrhythmia", "stroke", "gastritis", "gerd", "renal failure"
    ],
    "MEDICATION": [
        "paracetamol", "acetaminophen", "ibuprofen", "amlodipine", "metformin",
        "aspirin", "azithromycin", "cetirizine", "omeprazole", "atorvastatin",
        "lisinopril", "losartan", "amoxicillin", "ciprofloxacin",
        "levothyroxine", "pantoprazole", "salbutamol", "albuterol",
        "hydrochlorothiazide", "prednisone"
    ],
    "TEST": [
        "cbc", "complete blood count", "blood test", "urine test", "urinalysis",
        "mri", "ct scan", "x-ray", "chest x-ray", "ecg", "ekg", "ultrasound",
        "blood pressure test", "lipid panel", "liver function test",
        "renal function test", "blood glucose test", "hba1c", "echocardiogram"
    ],
    "PROCEDURE": [
        "surgery", "biopsy", "endoscopy", "colonoscopy", "vaccination",
        "physiotherapy", "intubation", "dialysis", "appendectomy", "catheterization"
    ],
    "BODY_PART": [
        "heart", "lungs", "liver", "kidney", "kidneys", "stomach", "brain",
        "chest", "abdomen", "throat", "arm", "leg", "head", "neck",
        "back", "shoulder", "knee", "eye", "ear"
    ]
}


class MedicalEntityExtractor:
    """Extracts medical entities from preprocessed clinical text using spaCy EntityRuler."""

    def __init__(self, vocabulary: Optional[Dict[str, List[str]]] = None):
        self.vocab = self._normalize_vocabulary(vocabulary or config.nlp_vocabulary or DEFAULT_VOCABULARY)
        self.nlp = self._initialize_spacy_pipeline()

    def _normalize_vocabulary(self, vocab_in: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Normalize vocabulary keys to uppercase category names."""
        normalized: Dict[str, List[str]] = {}
        for key, items in vocab_in.items():
            cat_key = key.upper()
            if cat_key == "SYMPTOMS":
                cat_key = "SYMPTOM"
            elif cat_key == "CONDITIONS":
                cat_key = "CONDITION"
            elif cat_key == "MEDICATIONS":
                cat_key = "MEDICATION"
            elif cat_key == "TESTS":
                cat_key = "TEST"
            elif cat_key == "PROCEDURES":
                cat_key = "PROCEDURE"
            elif cat_key == "BODY_PARTS":
                cat_key = "BODY_PART"
            normalized[cat_key] = items
        return normalized

    def _initialize_spacy_pipeline(self) -> spacy.language.Language:
        """Initialize spaCy pipeline with customized EntityRuler rules."""
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(
                f"Could not load 'en_core_web_sm': {e}. Creating a blank 'en' model. "
                "For full POS/syntax capabilities, run: python -m spacy download en_core_web_sm"
            )
            nlp = spacy.blank("en")

        # Configure EntityRuler with phrase matcher matching case-insensitively
        ruler_config = {"overwrite_ents": True, "phrase_matcher_attr": "LOWER"}
        
        # Check if entity_ruler already in pipeline
        if "entity_ruler" in nlp.pipe_names:
            nlp.remove_pipe("entity_ruler")

        # Insert before ner if ner exists, otherwise at the end
        if "ner" in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner", config=ruler_config)
        else:
            ruler = nlp.add_pipe("entity_ruler", config=ruler_config)

        # Generate patterns sorted by length descending so longer phrases match first
        patterns = []
        for category, terms in self.vocab.items():
            sorted_terms = sorted(terms, key=len, reverse=True)
            for term in sorted_terms:
                patterns.append({"label": category, "pattern": term})

        ruler.add_patterns(patterns)
        return nlp

    def _deduplicate_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Deduplicate entities case-insensitively, preserving original casing and order."""
        seen: Set[Tuple[str, EntityStatus]] = set()
        deduped: List[MedicalEntity] = []

        for ent in entities:
            key = (ent.text.lower(), ent.status)
            if key not in seen:
                seen.add(key)
                deduped.append(ent)

        return deduped

    def extract(self, text: str) -> ExtractedMedicalData:
        """Run the end-to-end traditional NLP extraction pipeline on raw or preprocessed text.
        
        Args:
            text: Input clinical document text
            
        Returns:
            ExtractedMedicalData: Validated Pydantic model containing all extracted elements
        """
        clean_text, sentences = preprocess_text(text)
        if not clean_text or not sentences:
            return ExtractedMedicalData(raw_text_length=0)

        symptoms: List[MedicalEntity] = []
        conditions: List[MedicalEntity] = []
        medications: List[MedicalEntity] = []
        tests: List[MedicalEntity] = []
        procedures: List[MedicalEntity] = []
        body_parts: List[MedicalEntity] = []

        target_categories = {"SYMPTOM", "CONDITION", "MEDICATION", "TEST", "PROCEDURE", "BODY_PART"}

        # Process each sentence to maintain local syntactic boundaries
        for sent_text in sentences:
            doc = self.nlp(sent_text)
            for ent in doc.ents:
                cat = ent.label_
                if cat not in target_categories:
                    continue

                # Detect status (PRESENT, NOT_PRESENT, HISTORICAL)
                status = negation_detector.detect_status(ent.text, sent_text)

                entity_obj = MedicalEntity(
                    text=ent.text,
                    category=cat,
                    status=status,
                    sentence=sent_text,
                    start_char=ent.start_char,
                    end_char=ent.end_char
                )

                if cat == "SYMPTOM":
                    symptoms.append(entity_obj)
                elif cat == "CONDITION":
                    conditions.append(entity_obj)
                elif cat == "MEDICATION":
                    medications.append(entity_obj)
                elif cat == "TEST":
                    tests.append(entity_obj)
                elif cat == "PROCEDURE":
                    procedures.append(entity_obj)
                elif cat == "BODY_PART":
                    body_parts.append(entity_obj)

        # Deduplicate entities
        clean_symptoms = self._deduplicate_entities(symptoms)
        clean_conditions = self._deduplicate_entities(conditions)
        clean_medications = self._deduplicate_entities(medications)
        clean_tests = self._deduplicate_entities(tests)
        clean_procedures = self._deduplicate_entities(procedures)
        clean_body_parts = self._deduplicate_entities(body_parts)

        # Extract dosages and correlate with medications
        known_med_names = [m.text for m in clean_medications]
        dosages = dosage_extractor.extract_dosages(sentences, known_medications=known_med_names)

        # Extract clinical findings
        findings = findings_extractor.extract_findings(sentences)

        return ExtractedMedicalData(
            symptoms=clean_symptoms,
            conditions=clean_conditions,
            medications=clean_medications,
            dosages=dosages,
            tests=clean_tests,
            procedures=clean_procedures,
            body_parts=clean_body_parts,
            important_findings=findings,
            raw_text_length=len(clean_text)
        )


# Default singleton instance
entity_extractor = MedicalEntityExtractor()
