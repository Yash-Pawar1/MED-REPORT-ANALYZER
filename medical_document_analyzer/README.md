# Medical Document NLP Analyzer & Report Generator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-76%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **CRITICAL MEDICAL & SAFETY DISCLAIMER**:  
> This application is an **information extraction and document summarization system** developed for educational, academic, and research demonstration. It is **NOT** a clinical decision-support system, **NOT** a diagnostic tool, **NOT** a treatment recommender, and **NOT** a medical advice chatbot. Never use this system for real-world patient diagnoses or clinical decision making. Real clinical patient records must not be uploaded without appropriate authorization and HIPAA/GDPR safeguards.

---

## 1. Project Overview & Problem Statement

Clinical documents (progress notes, discharge summaries, laboratory reports) contain dense medical terminology, dosage prescriptions, test orders, and critical negations. Extracting key clinical facts and generating clear executive summaries is essential for clinical workflows.

However, relying entirely on black-box Large Language Models (LLMs) to perform direct raw medical entity extraction introduces severe risks of **hallucination**, omission of subtle negations (e.g. converting *"denies fever"* into *"has fever"*), or unauthorized clinical inferences.

### The Solution: Two-Tier Architecture
This project implements a strict, safety-first pipeline where:
1. **Traditional NLP & Rule-Based Processing** performs **100% of the initial medical entity extraction**, negation detection, dosage parsing, and finding extraction.
2. **Pydantic Validation Models** enforce strict schema typing and validation on extracted facts.
3. **Cloud LLM (Google Gemini or OpenAI)** receives **ONLY the structured validated data** (never raw clinical documents or unverified patient text) to synthesize a concise, structured factual summary.
4. **Defensive Output Validation** inspects the generated report for mandatory sections, absence of diagnostic claims, and consistency against extracted facts.
5. **Deterministic Fallback Engine** automatically generates a rule-based fallback report if cloud APIs fail, timeout, or return malformed outputs.

---

## 2. Core Architecture Pipeline

```
          ┌──────────────────────────────────────────────┐
          │     Clinical Document Upload (PDF / TXT)     │
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────┐
          │       Safe Ingestion & Text Extraction       │
          │     PyMuPDF (Page-by-page) / UTF-8 TXT       │
          │        (Scanned PDF Detection Check)         │
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────┐
          │            NLP Text Preprocessing            │
          │   Unicode NFKC / Whitespace / Segmentation   │
          │   (Strict preservation of negation words)    │
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────┐
          │      Traditional NLP Entity Extraction       │
          │        spaCy EntityRuler + Regex Rules       │
          │  Symptoms, Conditions, Medications, Tests,   │
          │  Procedures, Body Parts, Findings, Dosages   │
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────┐
          │       Negation & Temporal Status Engine      │
          │  PRESENT | NOT_PRESENT | HISTORICAL | UNKNOWN│
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────┐
          │     Structured Data Model (Pydantic v2)      │
          │         ExtractedMedicalData Schema          │
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────┐
          │            Cloud LLM Summarization           │
          │   Google Gemini API  OR  OpenAI API ONLY     │
          │  (LLM receives ONLY validated structured data│
          │    never raw documents or unverified text)   │
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────┐
          │           Defensive Output Validation        │
          │    12-Section Check / Hallucination Guard    │
          │       (Fallback to Rule-Based Report)        │
          └──────────────────────┬───────────────────────┘
                                 │
                                 ▼
          ┌──────────────────────────────────────────────┐
          │          Streamlit Interactive UI            │
          │   Entity Inspection Cards / Export to TXT    │
          └──────────────────────────────────────────────┘
```

---

## 3. Key Features

- **Document Ingestion**:
  - Native page-by-page PDF extraction using `PyMuPDF` (`pymupdf`).
  - Scanned/image-only PDF detection (informs user that OCR is not enabled).
  - Robust UTF-8 text decoding with fallback for UTF-8-SIG and Latin-1.
  - File extension and 10 MB size limit validation.
- **Rule-Based Traditional NLP**:
  - `spaCy` `EntityRuler` dictionary matching across 6 core categories.
  - Extensible vocabulary loaded via `config/config.yaml`.
  - Case-insensitive entity deduplication preserving initial casing.
- **Negation & Historical Engine**:
  - Context window analysis with clause boundary awareness (`but`, `however`, `;`).
  - Distinguishes `PRESENT`, `NOT_PRESENT` (denied/absent), and `HISTORICAL` (prior history).
- **Dosage & Frequency Parsing**:
  - Regex detection of clinical units (`mg`, `ml`, `tablets`, `mcg`, `mg/day`, etc.).
  - Frequency extraction (`twice daily`, `every 8 hours`, `before meals`, etc.).
  - Intra-sentence medication association (prevents distant cross-sentence bleeding).
  - Strict exclusion of arbitrary non-clinical numbers.
- **Verbatim Finding Extraction**:
  - Extracts sentences matching clinical indicator keywords (`elevated`, `abnormal`, `prescribed`, `recommended`, etc.) preserving source wording.
- **Cloud LLM Abstraction**:
  - Clean polymorphic interface (`BaseLLMClient`, `GeminiClient`, `OpenAIClient`).
  - Configurable switching via `LLM_PROVIDER=gemini` or `LLM_PROVIDER=openai`.
  - Non-intrusive safe API connection test (tests ping without uploading documents).
- **Resilience & Fallback**:
  - Deterministic Rule-Based Fallback Report generated automatically if API credentials are missing, network times out, or output fails validation.
  - 100% of NLP extraction is preserved regardless of LLM status.
- **Exporting**:
  - Download full report as `medical_document_report.txt` with audit metadata and disclaimer.

---

## 4. Technology Stack

- **Python**: 3.11+
- **Frontend / UI**: Streamlit 1.30+
- **NLP & Parsing**:
  - spaCy 3.7+ (`en_core_web_sm` + `EntityRuler`)
  - NLTK 3.8+ (Sentence segmentation)
  - PyMuPDF 1.23+ (`pymupdf`)
  - Python standard `regex` and `unicodedata`
- **Data Validation**: Pydantic v2
- **Configuration & Environment**: PyYAML, python-dotenv
- **Cloud LLM SDKs**:
  - Official Google Gemini SDK (`google-genai`)
  - Official OpenAI SDK (`openai`)
- **Testing**: pytest (76 automated unit/integration tests with mocks)

---

## 5. Project Directory Structure

```
medical_document_analyzer/
│
├── app.py                             # Streamlit interactive application
├── requirements.txt                   # Production dependencies
├── README.md                          # Comprehensive documentation
├── PROJECT_STATE.md                   # Development state, verification & roadmap
├── .gitignore                         # Git exclusion rules for .env and caches
├── .env.example                       # Template for API credentials and configs
├── pytest.ini                         # Test runner configuration
│
├── prompts/
│   └── medical_summary_prompt.txt     # Strict summarization prompt template
│
├── config/
│   └── config.yaml                    # Non-secret settings, vocabulary & regexes
│
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── config.py                      # Centralized configuration loader
│   ├── models.py                      # Pydantic data schemas & Enums
│   ├── pdf_processor.py               # PDF/TXT extraction & scanned PDF detection
│   ├── text_processor.py              # Unicode & whitespace normalization
│   ├── entity_extractor.py            # spaCy EntityRuler traditional extraction
│   ├── negation.py                    # Clause-bounded negation & history detector
│   ├── dosage_extractor.py            # Regex dosage parsing & medication correlation
│   ├── findings_extractor.py          # Clinical finding sentence extraction
│   ├── llm_client.py                  # Gemini and OpenAI client abstraction
│   ├── llm_report_generator.py        # Prompt construction & generation orchestrator
│   ├── report_validator.py            # 12-section validation & hallucination guard
│   ├── report_formatter.py            # Fallback report builder & download formatter
│   └── utils.py                       # Sanitized logging and path helpers
│
├── tests/
│   ├── __init__.py
│   ├── test_pdf_processor.py          # Tests PDF, TXT, scanned checks, and errors
│   ├── test_text_processor.py         # Tests normalization & negation token preservation
│   ├── test_entity_extractor.py       # Tests traditional entity extraction & dedup
│   ├── test_negation.py               # Tests affirmative, negation, history, boundaries
│   ├── test_dosage_extractor.py       # Tests dosage units, frequencies, association
│   ├── test_findings_extractor.py     # Tests verbatim clinical finding sentences
│   ├── test_models.py                 # Tests Pydantic models & payload serialization
│   ├── test_llm_client.py             # Tests Gemini/OpenAI clients with mock SDKs
│   ├── test_report_generator.py       # Tests LLM orchestration & fallback triggers
│   ├── test_report_validator.py       # Tests required sections & prohibited phrases
│   └── test_end_to_end.py             # Full pipeline test on sample medical note
│
└── data/
    └── sample_medical_note.txt        # Academic benchmark clinical note
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Git

### Step 1: Clone or Navigate to the Repository
```bash
cd medical_document_analyzer
```

### Step 2: Create and Activate Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt / PowerShell)
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

---

## 7. API Configuration & Environment Setup

Copy `.env.example` to `.env`:
```bash
cp .env.example .env     # macOS/Linux
copy .env.example .env   # Windows
```

Open `.env` and configure your chosen provider and credentials:

```ini
# ============================================================
# LLM Provider: "gemini" or "openai"
# ============================================================
LLM_PROVIDER=gemini

# ============================================================
# Google Gemini API Configuration
# Get your API key from: https://aistudio.google.com/
# ============================================================
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# ============================================================
# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
# ============================================================
OPENAI_API_KEY=your_actual_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# ============================================================
# Operational Settings
# ============================================================
LLM_TIMEOUT=30
MAX_FILE_SIZE_MB=10
```

> **Security Note**: Never commit `.env` to version control. The `.gitignore` file is configured to exclude all `.env` files automatically.

---

## 8. Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Using the Interface:
1. Select your LLM Provider (**Gemini** or **OpenAI**) in the sidebar.
2. Click **Test API Connection** to verify your API credentials safely.
3. Upload a clinical document (`.pdf` or `.txt`) using the file uploader.
4. Click **🚀 Analyze Document**.
5. Inspect:
   - Extracted document text in the expandable viewer.
   - Traditional NLP extracted entities categorized across Symptoms, Conditions, Medications, Dosages, Tests, Procedures, Body Parts, and Findings.
   - The synthesized factual Medical Information Summary Report.
6. Click **💾 Download Report as TXT** to export the verified report.

---

## 9. Running Automated Tests

All tests run completely offline with mocked API responses. **Zero internet connection or API keys are required to run tests.**

Run the complete test suite:
```bash
python -m pytest -v
```

Expected output:
```
============================= test session starts =============================
platform win32 -- Python 3.12.8, pytest-9.1.1
collected 76 items

tests/test_dosage_extractor.py ........                                  [ 10%]
tests/test_end_to_end.py .                                               [ 11%]
tests/test_entity_extractor.py ........                                  [ 22%]
tests/test_findings_extractor.py ....                                    [ 27%]
tests/test_llm_client.py ..............                                  [ 46%]
tests/test_models.py ....                                                [ 51%]
tests/test_negation.py .........                                         [ 63%]
tests/test_pdf_processor.py ............                                 [ 79%]
tests/test_report_generator.py ....                                      [ 84%]
tests/test_report_validator.py .....                                     [ 91%]
tests/test_text_processor.py .......                                     [100%]

============================= 76 passed in 11.51s =============================
```

---

## 10. Sample Input & Expected Extraction

### Sample Clinical Note (`data/sample_medical_note.txt`):
```text
Patient presented with fever, persistent cough and fatigue for three days. The patient has a history of hypertension. Paracetamol 500 mg was prescribed for symptomatic relief. A CBC and chest X-ray were recommended. Blood pressure was elevated during the examination. The patient denies chest pain.
```

### Traditional NLP Extraction:
- **Symptoms**:
  - `fever` (`PRESENT`)
  - `persistent cough` (`PRESENT`)
  - `fatigue` (`PRESENT`)
  - `chest pain` (`NOT_PRESENT` / denied)
- **Medical Conditions**:
  - `hypertension` (`HISTORICAL`)
- **Medications**:
  - `Paracetamol`
- **Dosages**:
  - `500 mg` (associated with `Paracetamol`)
- **Diagnostic Tests**:
  - `CBC`
  - `chest X-ray`
- **Body Parts**:
  - `chest`
- **Important Findings**:
  - *"Blood pressure was elevated during the examination."* (Trigger: `elevated`)
  - *"Paracetamol 500 mg was prescribed for symptomatic relief."* (Trigger: `prescribed`)
  - *"A CBC and chest X-ray were recommended."* (Trigger: `recommended`)

### Expected Synthesized Report:
```
MEDICAL DOCUMENT SUMMARY

Symptoms:
- Fever
- Persistent cough
- Fatigue
- Chest pain (denied / not present)

Medical Conditions:
- Hypertension (historical)

Medications:
- Paracetamol

Dosages:
- 500 mg (for Paracetamol)

Tests:
- CBC
- Chest X-ray

Procedures:
- Not mentioned.

Body Parts:
- Chest

Important Findings:
- Blood pressure was elevated during the examination.
- Paracetamol 500 mg was prescribed for symptomatic relief.
- A CBC and chest X-ray were recommended.

Brief Summary:
- The patient presented with fever, persistent cough, and fatigue.
- A history of hypertension was noted, and blood pressure was elevated upon examination.
- Paracetamol 500 mg was prescribed for symptomatic relief, with a CBC and chest X-ray recommended.
- The patient explicitly denied chest pain.

Potential Missing Information:
- Surgical procedures, follow-up timeline, and dosage frequency were not mentioned.

Safety Note:
- This report is generated from extracted document information and is not medical advice.
```

---

## 11. Error Handling Matrix

| Scenario / Case | System Response & UI Handling |
| :--- | :--- |
| **No file selected** | Displays friendly informational banner prompting user to upload a file. |
| **Unsupported file type** (e.g. `.docx`, `.exe`) | Rejects upload with clear error: *"Unsupported file format. Only .pdf and .txt files are supported."* |
| **Empty document** (0 bytes or whitespace) | Catches `EmptyDocumentError` and displays clean error notification. |
| **Corrupted PDF** | Catches `CorruptedPDFError` and displays clean notification without stack traces. |
| **Scanned / Image-only PDF** | Detects lack of text alongside image content; alerts user: *"This PDF appears to contain scanned/image-only content. OCR is not enabled in this version."* |
| **No medical entities found** | Informs user that no entities were detected; generates graceful fallback summary. |
| **Missing API key** | Displays warning banner explaining how to configure credentials in `.env`; activates Rule-Based Fallback Report. |
| **API Timeout / Network error** | Handles timeout gracefully, preserves all NLP extractions, and generates Rule-Based Fallback Report. |
| **Malformed / Hallucinatory LLM response** | `ReportValidator` catches missing sections or diagnostic terms; automatically presents verified Rule-Based Fallback Report. |

---

## 12. Security & Privacy Guarantees

1. **Zero Secret Leakage**: API keys are loaded solely through environment variables and `.env`. Keys are never logged, displayed in the UI, or written into export files.
2. **Data Minimization**: The raw clinical document is **never** sent to the Cloud LLM. Only structured, non-identifying entity lists are transmitted.
3. **No Persistent Patient Storage**: Documents and generated reports exist solely in memory during session execution and are not stored in databases.
4. **Sanitized Logging**: Loggers output technical and pipeline milestones only, completely omitting patient content and payloads.

---

## 13. Limitations & Future Improvements

### Current Limitations:
- **OCR Disabled**: Scanned PDFs without embedded text streams are detected but cannot be read without an external OCR engine (such as Tesseract).
- **Rule-Based Vocabulary**: While comprehensive and extensible via `config.yaml`, out-of-vocabulary rare drug names or novel syndromes require pattern updates.
- **English Language**: Optimized for clinical English documents.

### Future Roadmap:
- Opt-in local OCR engine integration (e.g., Tesseract or DocTR).
- Expansion of vocabulary using UMLS (Unified Medical Language System) and SNOMED-CT dictionary tables.
- Multi-lingual clinical document preprocessing.
- FHIR (Fast Healthcare Interoperability Resources) JSON export formatting.
