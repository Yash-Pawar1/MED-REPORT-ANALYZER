# Medical Document NLP Analyzer & Report Generator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-76%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **CRITICAL MEDICAL & SAFETY DISCLAIMER**:  
> This application is an **information extraction and document summarization system** developed for educational, academic, and research demonstration. It is **NOT** a clinical decision-support system, **NOT** a diagnostic tool, **NOT** a treatment recommender, and **NOT** a medical advice chatbot. Never use this system for real-world patient diagnoses or clinical decision making. Real clinical patient records must not be uploaded without appropriate authorization and HIPAA/GDPR safeguards.

---

## Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download spaCy English model
python -m spacy download en_core_web_sm

# 3. Configure API keys (optional if testing with mocks)
copy .env.example .env

# 4. Run automated test suite (76 tests pass offline)
python -m pytest -v

# 5. Launch the Streamlit application
streamlit run app.py
```

For full architecture details, configuration options, and module descriptions, please refer to the documentation inside `medical_document_analyzer/README.md`.
