"""Document extraction engine supporting PDF and TXT medical files.

Uses PyMuPDF (fitz) for page-by-page PDF extraction with scanned/image detection.
Safely handles encoding, corrupted streams, and file limits.
"""

import io
from pathlib import Path
from typing import BinaryIO, Union
import pymupdf  # PyMuPDF official import

from src.utils import setup_logger

logger = setup_logger(__name__)


class DocumentProcessingError(Exception):
    """Base exception for document processing issues."""
    pass


class UnsupportedFormatError(DocumentProcessingError):
    """Raised when the uploaded file extension is not supported."""
    pass


class EmptyDocumentError(DocumentProcessingError):
    """Raised when an uploaded document contains zero bytes or no text."""
    pass


class ScannedPDFError(DocumentProcessingError):
    """Raised when a PDF contains only scanned images with no extractable text."""
    pass


class CorruptedPDFError(DocumentProcessingError):
    """Raised when a PDF file is damaged, password protected, or corrupted."""
    pass


def validate_file_metadata(filename: str, file_size_bytes: int, max_size_mb: int = 10) -> None:
    """Validate file extension and size constraints.
    
    Raises:
        UnsupportedFormatError: If extension is not .pdf or .txt
        EmptyDocumentError: If file is 0 bytes
        DocumentProcessingError: If file exceeds maximum size
    """
    if not filename:
        raise UnsupportedFormatError("No filename provided.")

    ext = Path(filename).suffix.lower()
    if ext not in [".pdf", ".txt"]:
        raise UnsupportedFormatError(
            f"Unsupported file format '{ext}'. Only .pdf and .txt files are supported."
        )

    if file_size_bytes == 0:
        raise EmptyDocumentError("The uploaded file is empty (0 bytes).")

    max_bytes = max_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise DocumentProcessingError(
            f"File size ({file_size_bytes / (1024 * 1024):.1f} MB) exceeds maximum allowed size of {max_size_mb} MB."
        )


def extract_text_from_txt(content: Union[bytes, str, BinaryIO]) -> str:
    """Safely decode text from a TXT document.
    
    Supports UTF-8 with fallbacks for UTF-8-sig and Latin-1.
    """
    if isinstance(content, str):
        text = content
    elif isinstance(content, (bytes, bytearray)):
        if len(content) == 0:
            raise EmptyDocumentError("The text document is empty.")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                try:
                    text = content.decode("latin-1")
                except Exception as e:
                    logger.error(f"Failed to decode text file: {e}")
                    raise DocumentProcessingError("Could not decode text file with supported encodings.")
    elif hasattr(content, "read"):
        raw_bytes = content.read()
        return extract_text_from_txt(raw_bytes)
    else:
        raise DocumentProcessingError("Unsupported content source type for TXT extraction.")

    cleaned = text.strip()
    if not cleaned:
        raise EmptyDocumentError("The text document contains no readable characters.")
    return cleaned


def extract_text_from_pdf(content: Union[bytes, BinaryIO, str, Path]) -> str:
    """Extract readable text from a PDF file page by page using PyMuPDF.
    
    Detects scanned or image-only documents and preserves paragraph separation.
    """
    doc = None
    try:
        if isinstance(content, (str, Path)):
            path_obj = Path(content)
            if not path_obj.exists():
                raise DocumentProcessingError(f"File not found: {path_obj}")
            if path_obj.stat().st_size == 0:
                raise EmptyDocumentError("The PDF file is empty (0 bytes).")
            doc = pymupdf.open(str(path_obj))
        elif isinstance(content, (bytes, bytearray)):
            if len(content) == 0:
                raise EmptyDocumentError("The PDF stream is empty (0 bytes).")
            doc = pymupdf.open(stream=content, filetype="pdf")
        elif hasattr(content, "read"):
            data = content.read()
            if len(data) == 0:
                raise EmptyDocumentError("The PDF stream is empty (0 bytes).")
            doc = pymupdf.open(stream=data, filetype="pdf")
        else:
            raise DocumentProcessingError("Unsupported content source type for PDF extraction.")

        if doc.is_encrypted:
            raise CorruptedPDFError("PDF is encrypted or password-protected and cannot be read.")

        page_texts = []
        has_images = False
        total_pages = len(doc)

        if total_pages == 0:
            raise EmptyDocumentError("The PDF contains 0 pages.")

        for page_idx in range(total_pages):
            page = doc[page_idx]
            text = page.get_text("text").strip()
            if text:
                page_texts.append(text)
            
            # Check if page contains images
            if len(page.get_images()) > 0:
                has_images = True

        combined_text = "\n\n".join(page_texts).strip()

        # Scanned PDF detection:
        if not combined_text:
            if has_images:
                raise ScannedPDFError(
                    "This PDF appears to contain scanned/image-only content. OCR is not enabled in this version."
                )
            else:
                raise EmptyDocumentError("Could not extract readable text from this document.")

        return combined_text

    except (ScannedPDFError, EmptyDocumentError, CorruptedPDFError, UnsupportedFormatError):
        raise
    except pymupdf.FileDataError as e:
        logger.error(f"PyMuPDF FileDataError: {e}")
        raise CorruptedPDFError("The PDF file is corrupted or not a valid PDF document.")
    except Exception as e:
        logger.error(f"Unexpected error processing PDF: {e}")
        raise DocumentProcessingError(f"Failed to extract text from PDF: {str(e)}")
    finally:
        if doc is not None:
            doc.close()


def extract_document_text(
    filename: str,
    content: Union[bytes, BinaryIO, str, Path],
    max_size_mb: int = 10
) -> str:
    """Unified entry point for document text extraction with metadata validation."""
    # Estimate size if bytes/file
    size = 0
    if isinstance(content, (bytes, bytearray)):
        size = len(content)
    elif isinstance(content, (str, Path)):
        p = Path(content)
        size = p.stat().st_size if p.exists() else 0
    elif hasattr(content, "getbuffer"):
        size = content.getbuffer().nbytes
    elif hasattr(content, "tell") and hasattr(content, "seek"):
        curr = content.tell()
        content.seek(0, 2)
        size = content.tell()
        content.seek(curr)

    validate_file_metadata(filename, size, max_size_mb)

    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(content)
    elif ext == ".txt":
        return extract_text_from_txt(content)
    else:
        raise UnsupportedFormatError(f"Unsupported extension: '{ext}'")
