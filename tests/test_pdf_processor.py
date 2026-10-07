"""Tests for PDF and TXT document text extraction and validation."""

import io
import pytest
import pymupdf

from src.pdf_processor import (
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_document_text,
    validate_file_metadata,
    UnsupportedFormatError,
    EmptyDocumentError,
    ScannedPDFError,
    CorruptedPDFError,
    DocumentProcessingError,
)


def create_sample_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper to create an in-memory PDF with specified text per page."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


from PIL import Image


def create_scanned_mock_pdf_bytes() -> bytes:
    """Helper to create a PDF with an image and no text."""
    doc = pymupdf.open()
    page = doc.new_page()
    img = Image.new("RGB", (64, 64), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(pymupdf.Rect(50, 50, 150, 150), stream=buf.getvalue())
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestDocumentValidation:
    """Test metadata, file size, and extension validation."""

    def test_supported_extensions(self):
        # Should not raise
        validate_file_metadata("note.txt", 100, 10)
        validate_file_metadata("report.pdf", 100, 10)

    def test_unsupported_extensions(self):
        with pytest.raises(UnsupportedFormatError):
            validate_file_metadata("malicious.exe", 100, 10)
        with pytest.raises(UnsupportedFormatError):
            validate_file_metadata("data.docx", 100, 10)

    def test_empty_file_metadata(self):
        with pytest.raises(EmptyDocumentError):
            validate_file_metadata("empty.txt", 0, 10)

    def test_file_size_limit_exceeded(self):
        too_large_bytes = 11 * 1024 * 1024  # 11 MB
        with pytest.raises(DocumentProcessingError, match="exceeds maximum"):
            validate_file_metadata("huge.pdf", too_large_bytes, max_size_mb=10)


class TestTxtExtraction:
    """Test text extraction from TXT files."""

    def test_extract_valid_utf8_txt(self):
        sample = "Patient reported fever and cough."
        result = extract_text_from_txt(sample.encode("utf-8"))
        assert "fever and cough" in result

    def test_extract_unicode_txt(self):
        sample = "Temperature was 38.5 °C with fatigue."
        result = extract_text_from_txt(sample.encode("utf-8"))
        assert "38.5 °C" in result

    def test_empty_txt_raises_error(self):
        with pytest.raises(EmptyDocumentError):
            extract_text_from_txt(b"")

    def test_whitespace_only_txt_raises_error(self):
        with pytest.raises(EmptyDocumentError):
            extract_text_from_txt(b"   \n\t  \n  ")


class TestPdfExtraction:
    """Test PyMuPDF extraction, multi-page, scanned detection, and corruption."""

    def test_valid_single_page_pdf(self):
        pdf_bytes = create_sample_pdf_bytes(["Patient presented with hypertension and headache."])
        text = extract_text_from_pdf(pdf_bytes)
        assert "hypertension and headache" in text

    def test_valid_multipage_pdf(self):
        pages = [
            "Page 1: History of diabetes.",
            "Page 2: Prescribed Metformin 500 mg."
        ]
        pdf_bytes = create_sample_pdf_bytes(pages)
        text = extract_text_from_pdf(pdf_bytes)
        assert "Page 1: History of diabetes." in text
        assert "Page 2: Prescribed Metformin 500 mg." in text

    def test_scanned_pdf_detection(self):
        scanned_bytes = create_scanned_mock_pdf_bytes()
        with pytest.raises(ScannedPDFError, match="OCR is not enabled"):
            extract_text_from_pdf(scanned_bytes)

    def test_corrupted_pdf_raises_error(self):
        corrupted_bytes = b"%PDF-1.4\nCorrupted binary garbage not valid pdf..."
        with pytest.raises(CorruptedPDFError):
            extract_text_from_pdf(corrupted_bytes)

    def test_empty_pdf_stream_raises_error(self):
        with pytest.raises(EmptyDocumentError):
            extract_text_from_pdf(b"")

    def test_extract_document_text_dispatcher(self):
        txt_bytes = b"Patient has mild fever."
        res = extract_document_text("note.txt", txt_bytes)
        assert "Patient has mild fever" in res
