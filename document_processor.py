"""
Document Processor for College Regulations RAG Chatbot
Extracts text from PDF documents with page-level tracking and splits them
into semantically coherent, overlapping chunks with metadata.
"""

import os
import re
import hashlib
from typing import List, Dict, Any, Union, BinaryIO
from pypdf import PdfReader


class DocumentChunk:
    """Represents a text chunk with document provenance metadata."""
    def __init__(self, chunk_id: str, text: str, source: str, page_number: int, chunk_index: int):
        self.chunk_id = chunk_id
        self.text = text.strip()
        self.source = source
        self.page_number = page_number
        self.chunk_index = chunk_index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            source=data["source"],
            page_number=data["page_number"],
            chunk_index=data.get("chunk_index", 0)
        )


def clean_text(text: str) -> str:
    """Normalizes whitespace and cleans extracted PDF text."""
    if not text:
        return ""
    # Normalize unicode spaces and linebreaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Replace multiple empty lines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove control characters except tab and newline
    text = "".join(ch for ch in text if ch == '\n' or ch == '\t' or ord(ch) >= 32)
    # Replace multiple inline spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def extract_pages_from_pdf(file_input: Union[str, BinaryIO, bytes], filename: str = "document.pdf") -> List[Dict[str, Any]]:
    """
    Extracts text from each page of a PDF document using pypdf.
    Returns a list of dicts: [{'page_number': int, 'text': str, 'source': str}]
    """
    pages_data = []
    try:
        reader = PdfReader(file_input)
        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ValueError("The uploaded PDF has 0 pages or is empty.")

        for idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            cleaned = clean_text(raw_text)
            pages_data.append({
                "page_number": idx + 1,
                "text": cleaned,
                "source": filename,
                "raw_char_count": len(raw_text)
            })

    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF '{filename}': {str(e)}")

    return pages_data


def split_text_recursive(text: str, chunk_size: int = 600, chunk_overlap: int = 120) -> List[str]:
    """
    Recursively splits text using natural boundaries (\n\n, \n, ., ' ') to produce
    chunks of size ~chunk_size with specified overlap.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Priority separators
    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    
    def _split(txt: str, sep_idx: int) -> List[str]:
        if len(txt) <= chunk_size or sep_idx >= len(separators):
            return [txt] if txt.strip() else []

        sep = separators[sep_idx]
        parts = txt.split(sep)
        chunks = []
        current = ""

        for part in parts:
            part_str = part + (sep if sep != " " else " ")
            if len(current) + len(part_str) <= chunk_size:
                current += part_str
            else:
                if current.strip():
                    chunks.append(current.strip())
                # If a single part is larger than chunk_size, split by next separator
                if len(part_str) > chunk_size:
                    sub_chunks = _split(part, sep_idx + 1)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    # Apply overlap from previous chunk if possible
                    if current and chunk_overlap > 0:
                        overlap_seed = current[-chunk_overlap:]
                        current = overlap_seed + part_str
                    else:
                        current = part_str

        if current.strip():
            chunks.append(current.strip())

        return chunks

    raw_chunks = _split(text, 0)
    
    # Filter out empty or trivially short chunks
    valid_chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 20]
    return valid_chunks


def process_document(file_input: Union[str, BinaryIO, bytes], 
                     filename: str, 
                     chunk_size: int = 600, 
                     chunk_overlap: int = 120) -> List[DocumentChunk]:
    """
    Extracts pages and generates chunk objects with source and page metadata.
    """
    pages = extract_pages_from_pdf(file_input, filename)
    all_chunks: List[DocumentChunk] = []

    file_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()[:6]
    total_chunk_idx = 0

    for page in pages:
        p_num = page["page_number"]
        p_text = page["text"]
        if not p_text.strip():
            continue

        text_chunks = split_text_recursive(p_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for sub_idx, chunk_text in enumerate(text_chunks):
            chunk_id = f"{file_hash}_p{p_num}_c{sub_idx}"
            chunk_obj = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source=filename,
                page_number=p_num,
                chunk_index=total_chunk_idx
            )
            all_chunks.append(chunk_obj)
            total_chunk_idx += 1

    return all_chunks
