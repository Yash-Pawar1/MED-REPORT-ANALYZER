"""Streamlit Web Application for Medical Document NLP Analyzer & Report Generator.

Provides an intuitive UI for uploading clinical documents (PDF/TXT),
inspecting traditional NLP entity extraction, running Cloud LLM summarization,
and downloading factual reports with strict safety guarantees.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so imports resolve seamlessly
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st

from src.config import config
from src.pdf_processor import (
    extract_document_text,
    DocumentProcessingError,
    UnsupportedFormatError,
    EmptyDocumentError,
    ScannedPDFError,
    CorruptedPDFError,
)
from src.entity_extractor import entity_extractor
from src.llm_client import get_llm_client, BaseLLMClient, LLMClientError, LLMAuthenticationError
from src.llm_report_generator import llm_report_generator
from src.report_formatter import format_export_download_text
from src.models import ExtractedMedicalData, EntityStatus
from src.utils import setup_logger

logger = setup_logger(__name__)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Medical Document NLP Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


def render_sidebar():
    """Render the sidebar with provider configuration and project info."""
    st.sidebar.title("⚙️ LLM Configuration")
    
    # Provider selection
    provider_options = ["gemini", "openai"]
    default_idx = provider_options.index(config.llm_provider) if config.llm_provider in provider_options else 0
    selected_provider = st.sidebar.selectbox(
        "LLM Provider",
        options=provider_options,
        index=default_idx,
        help="Select between Google Gemini API and OpenAI API."
    )

    # Model input
    default_model = config.gemini_model if selected_provider == "gemini" else config.openai_model
    model_name = st.sidebar.text_input(
        "Model Name",
        value=default_model,
        help="Specify the target cloud model identifier."
    )

    # API Key check indicator (never displays key text)
    api_key_configured = bool(config.get_api_key_for_provider(selected_provider))
    if api_key_configured:
        st.sidebar.success(f"✓ {selected_provider.capitalize()} API Key detected")
    else:
        st.sidebar.warning(f"⚠️ {selected_provider.capitalize()} API Key missing in .env")

    # API Connection Test Button
    if st.sidebar.button("Test API Connection", use_container_width=True):
        with st.sidebar:
            with st.spinner("Testing connection..."):
                try:
                    client = get_llm_client(provider=selected_provider, model=model_name)
                    success, message = client.test_connection()
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"Connection test failed: {str(e)}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 About this Project")
    st.sidebar.info(
        "**Pipeline Architecture**:\n"
        "1. PyMuPDF / UTF-8 Text Extraction\n"
        "2. Traditional spaCy EntityRuler NLP\n"
        "3. Regex Dosage & Status Detection\n"
        "4. Validated Pydantic Structure\n"
        "5. Cloud LLM Summarization ONLY\n"
        "6. Defensive Output Validation"
    )

    st.sidebar.markdown("### ⚠️ Safety Disclaimer")
    st.sidebar.caption(
        "This application is an information extraction and summarization tool for academic purposes. "
        "It does NOT provide medical diagnoses, treatment recommendations, or clinical advice."
    )

    return selected_provider, model_name


def display_nlp_results(data: ExtractedMedicalData):
    """Render extracted entities in structured cards and metrics."""
    st.subheader("🔍 Traditional NLP Extracted Information")
    st.caption("Extracted using rule-based spaCy EntityRuler and regex before contacting any LLM.")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Symptoms Found", len(data.symptoms))
    with col2:
        st.metric("Conditions Found", len(data.conditions))
    with col3:
        st.metric("Medications", len(data.medications))
    with col4:
        st.metric("Dosages & Tests", len(data.dosages) + len(data.tests))

    # Entity categories in tabs
    tab_symp, tab_cond, tab_med, tab_tests, tab_other, tab_find = st.tabs([
        "Symptoms", "Conditions", "Medications & Dosages", "Tests & Procedures", "Body Parts", "Important Findings"
    ])

    with tab_symp:
        if not data.symptoms:
            st.info("No symptoms detected.")
        else:
            for s in data.symptoms:
                status_color = "green" if s.status == EntityStatus.PRESENT else "red" if s.status == EntityStatus.NOT_PRESENT else "orange"
                st.markdown(f"- **{s.text.capitalize()}** — :{status_color}[`{s.status.value}`]")

    with tab_cond:
        if not data.conditions:
            st.info("No medical conditions detected.")
        else:
            for c in data.conditions:
                status_color = "green" if c.status == EntityStatus.PRESENT else "red" if c.status == EntityStatus.NOT_PRESENT else "orange"
                st.markdown(f"- **{c.text.capitalize()}** — :{status_color}[`{c.status.value}`]")

    with tab_med:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Medications")
            if not data.medications:
                st.info("No medications detected.")
            else:
                for m in data.medications:
                    st.markdown(f"- **{m.text.capitalize()}**")
        with c2:
            st.markdown("##### Extracted Dosages")
            if not data.dosages:
                st.info("No dosages detected.")
            else:
                for d in data.dosages:
                    desc = f"- **{d.value}**"
                    if d.medication:
                        desc += f" (Medication: *{d.medication}*)"
                    if d.frequency:
                        desc += f" — Freq: *{d.frequency}*"
                    st.markdown(desc)

    with tab_tests:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Diagnostic Tests")
            if not data.tests:
                st.info("No tests detected.")
            else:
                for t in data.tests:
                    st.markdown(f"- **{t.text}**")
        with c2:
            st.markdown("##### Procedures")
            if not data.procedures:
                st.info("No procedures detected.")
            else:
                for p in data.procedures:
                    st.markdown(f"- **{p.text.capitalize()}**")

    with tab_other:
        st.markdown("##### Anatomical Locations / Body Parts")
        if not data.body_parts:
            st.info("No body parts detected.")
        else:
            for b in data.body_parts:
                st.markdown(f"- **{b.text.capitalize()}**")

    with tab_find:
        st.markdown("##### Verbatim Important Findings")
        if not data.important_findings:
            st.info("No finding sentences detected.")
        else:
            for f in data.important_findings:
                st.markdown(f'> "{f.sentence}"  \n*(Trigger: `{f.trigger_keyword}`)*')


def main():
    st.title("🩺 Medical Document NLP Analyzer & Report Generator")
    st.markdown(
        "Extract structured medical entities using traditional NLP, "
        "and generate a strictly factual, non-diagnostic AI summary report."
    )

    selected_provider, model_name = render_sidebar()

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload a Medical Document (PDF or TXT)",
        type=["pdf", "txt"],
        help="Upload a clinical note, lab report, or medical summary."
    )

    # Analyze Button
    analyze_clicked = st.button("🚀 Analyze Document", type="primary", use_container_width=True)

    if not analyze_clicked:
        if not uploaded_file:
            st.info("👆 Please upload a PDF or TXT medical document to begin analysis.")
        return

    if not uploaded_file:
        st.warning("Please select a file to upload before clicking Analyze.")
        return

    # Process Document
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()

    # Step 1: Document Text Extraction
    with st.spinner("Extracting text from document..."):
        try:
            raw_text = extract_document_text(
                filename=filename,
                content=file_bytes,
                max_size_mb=config.max_file_size_mb
            )
        except UnsupportedFormatError as e:
            st.error(f"❌ File Format Error: {str(e)}")
            return
        except EmptyDocumentError as e:
            st.error(f"❌ Empty Document: {str(e)}")
            return
        except ScannedPDFError as e:
            st.error(f"❌ Scanned PDF: {str(e)}")
            return
        except CorruptedPDFError as e:
            st.error(f"❌ Corrupted File: {str(e)}")
            return
        except DocumentProcessingError as e:
            st.error(f"❌ Document Error: {str(e)}")
            return
        except Exception as e:
            logger.error(f"Unexpected document error: {e}")
            st.error("An unexpected error occurred while reading the document.")
            return

    st.success(f"✓ Successfully extracted text from `{filename}` ({len(raw_text)} characters).")

    # Display Extracted Text Expander
    with st.expander("📄 Extracted Document Text", expanded=False):
        st.text_area("Source Text", value=raw_text, height=200, disabled=True)

    # Step 2: Traditional NLP Entity Extraction
    with st.spinner("Running traditional NLP entity extraction and negation analysis..."):
        try:
            extracted_data = entity_extractor.extract(raw_text)
        except Exception as e:
            logger.error(f"NLP entity extraction failure: {e}")
            st.error("Failed to extract medical entities from text.")
            return

    if extracted_data.total_entities_count() == 0:
        st.warning("⚠️ No relevant medical entities were detected in this document.")

    # Display NLP Results
    display_nlp_results(extracted_data)

    st.markdown("---")

    # Step 3: Cloud LLM Summary Generation
    st.subheader("📑 Medical Information Summary Report")
    
    with st.spinner(f"Generating summary report via {selected_provider.upper()} ({model_name})..."):
        try:
            # Instantiate client
            client = get_llm_client(provider=selected_provider, model=model_name)
        except Exception as e:
            client = None

        report_text, is_fallback, error_msg, validation = llm_report_generator.generate_report(
            data=extracted_data,
            client=client
        )

    # Handle UI notification for fallback or success
    if is_fallback:
        st.warning(
            "⚠️ **AI report generation failed. The extracted NLP information is still available.**\n\n"
            f"Reason: {error_msg or 'API Unavailable'}\n\n"
            "*A deterministic Rule-Based Fallback Report has been generated below based strictly on verified NLP data.*"
        )
    else:
        st.success(f"✓ Summary generated and validated successfully using {selected_provider.upper()}.")

    # Render Report
    st.text_area("Generated Report", value=report_text, height=450)

    # Validation feedback if any warnings
    if validation and validation.warnings:
        with st.expander("🛡️ Validation Warnings"):
            for w in validation.warnings:
                st.warning(w)

    # Step 4: Download Report Button
    export_content = format_export_download_text(
        report_text=report_text,
        filename=filename,
        provider_name=selected_provider,
        model_name=model_name,
        is_fallback=is_fallback
    )

    st.download_button(
        label="💾 Download Report as TXT",
        data=export_content,
        file_name="medical_document_report.txt",
        mime="text/plain",
        use_container_width=True
    )


if __name__ == "__main__":
    main()
