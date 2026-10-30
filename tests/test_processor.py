"""
Unit Tests for Document Processor
"""

import os
import pytest
from document_processor import extract_pages_from_pdf, split_text_recursive, process_document


SAMPLE_PDF_PATH = os.path.join("data", "sample_college_regulations.pdf")


def test_sample_pdf_exists():
    assert os.path.exists(SAMPLE_PDF_PATH), "Sample PDF should exist before running tests."


def test_extract_pages_from_pdf():
    pages = extract_pages_from_pdf(SAMPLE_PDF_PATH, filename="sample_college_regulations.pdf")
    assert len(pages) >= 3, f"Expected at least 3 pages, got {len(pages)}"
    
    # Check page 1 has attendance regulations
    p1_text = pages[0]["text"]
    assert "ATTENDANCE" in p1_text.upper() or "ACADEMIC" in p1_text.upper()
    assert "75%" in p1_text
    assert pages[0]["page_number"] == 1
    assert pages[0]["source"] == "sample_college_regulations.pdf"


def test_split_text_recursive():
    sample_text = (
        "Paragraph 1 discusses academic calendar and dates.\n\n"
        "Paragraph 2 discusses attendance criteria and the 75% rule.\n\n"
        "Paragraph 3 discusses examination procedures and re-evaluation fees."
    )
    chunks = split_text_recursive(sample_text, chunk_size=80, chunk_overlap=15)
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c) > 0


def test_process_document_metadata():
    pages = extract_pages_from_pdf(SAMPLE_PDF_PATH, filename="sample_college_regulations.pdf")
    total_pages = len(pages)
    
    chunks = process_document(SAMPLE_PDF_PATH, filename="sample_college_regulations.pdf", chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 0
    
    # Verify metadata fields
    for chunk in chunks:
        assert chunk.source == "sample_college_regulations.pdf"
        assert 1 <= chunk.page_number <= total_pages
        assert chunk.chunk_id is not None
        assert len(chunk.text) > 0
