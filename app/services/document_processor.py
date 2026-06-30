"""
Document processing pipeline.
Handles PDF parsing, metadata extraction, and intelligent chunking
for Sri Lankan tax acts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

from core.config import ACTS_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, TAX_TYPE_MAP


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


def _extract_metadata_from_filename(filename: str) -> dict:
    """
    Extract structured metadata from act PDF filenames.

    Examples:
        VAT_Act_No_14[E]_2002_(Consolidation_2025).pdf
        NBT_Act_No._03_2020_E.pdf
        BNG_Act_No._11_2023_E.pdf
        IRA_Cons_Act_-_2025_Changes.pdf
        24-2025_E.pdf
    """
    metadata = {
        "source_file": filename,
        "tax_type": "General Tax",
        "tax_type_code": "GENERAL",
        "act_name": filename.replace(".pdf", "").replace("_", " "),
        "act_number": "",
        "year": 0,
        "is_amendment": False,
        "is_consolidation": False,
        "language": "English",
    }

    for code, full_name in TAX_TYPE_MAP.items():
        if filename.upper().startswith(code):
            metadata["tax_type"] = full_name
            metadata["tax_type_code"] = code
            break

    act_no_match = re.search(r"No[._\s]*(\d+)", filename)
    if act_no_match:
        metadata["act_number"] = act_no_match.group(1)

    years = re.findall(r"((?:19|20)\d{2})", filename)
    if years:
        metadata["year"] = max(int(y) for y in years)

    if "Amd" in filename or "Amendment" in filename:
        metadata["is_amendment"] = True

    if "Consolidation" in filename or "Cons" in filename:
        metadata["is_consolidation"] = True

    if "[E]" in filename or "_E." in filename or "_E_" in filename:
        metadata["language"] = "English"

    return metadata


def _detect_section_from_text(text: str) -> str:
    """Try to identify the section/part number from chunk text."""
    patterns = [
        r"(?:Section|SECTION)\s+(\d+[A-Za-z]*)",
        r"(?:PART|Part)\s+([IVXLCDM]+|\d+)",
        r"(?:Schedule|SCHEDULE)\s+([IVXLCDM]+|\d+)",
        r"(?:Article|ARTICLE)\s+(\d+)",
        r"(?:Chapter|CHAPTER)\s+([IVXLCDM]+|\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:500])
        if match:
            return match.group(0)
    return ""


def _extract_tax_rates_from_text(text: str) -> List[str]:
    """Extract any tax rate percentages mentioned in the text."""
    rates = re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|per\s*cent|percent)", text, re.IGNORECASE)
    return list(set(rates))


def parse_pdf(pdf_path: Path) -> List[dict]:
    """Parse a PDF file using PyMuPDF and return pages with text and metadata."""
    pages = []
    doc = fitz.open(str(pdf_path))
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "text": text,
                "page_number": page_num + 1,
                "total_pages": len(doc),
            })
    doc.close()
    return pages


def process_file(pdf_path: Path) -> List[Document]:
    """Process a single PDF file into Documents with rich metadata."""
    file_metadata = _extract_metadata_from_filename(pdf_path.name)
    pages = parse_pdf(pdf_path)

    documents = []
    for page_data in pages:
        text = page_data["text"]
        section = _detect_section_from_text(text)
        tax_rates = _extract_tax_rates_from_text(text)

        metadata = {
            **file_metadata,
            "page_number": page_data["page_number"],
            "total_pages": page_data["total_pages"],
            "section": section,
            "tax_rates_mentioned": ", ".join(tax_rates) if tax_rates else "",
            "has_tax_rates": len(tax_rates) > 0,
        }
        documents.append(Document(text=text, metadata=metadata))

    return documents


def process_acts(acts_folder: Optional[Path] = None) -> List[Document]:
    """
    Process all PDF acts from the Acts folder into Documents
    with rich metadata for optimal vector store retrieval.
    """
    folder = acts_folder or ACTS_FOLDER
    all_documents = []

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {folder}")

    for pdf_path in pdf_files:
        file_metadata = _extract_metadata_from_filename(pdf_path.name)
        pages = parse_pdf(pdf_path)

        for page_data in pages:
            text = page_data["text"]
            section = _detect_section_from_text(text)
            tax_rates = _extract_tax_rates_from_text(text)

            metadata = {
                **file_metadata,
                "page_number": page_data["page_number"],
                "total_pages": page_data["total_pages"],
                "section": section,
                "tax_rates_mentioned": ", ".join(tax_rates) if tax_rates else "",
                "has_tax_rates": len(tax_rates) > 0,
            }
            all_documents.append(Document(text=text, metadata=metadata))

    return all_documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into optimally-sized chunks using sentence-aware splitting.
    Preserves metadata across chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.text)
        for chunk in chunks:
            section = _detect_section_from_text(chunk)
            tax_rates = _extract_tax_rates_from_text(chunk)

            metadata = dict(doc.metadata)
            if section:
                metadata["section"] = section
            if tax_rates:
                metadata["tax_rates_mentioned"] = ", ".join(tax_rates)
                metadata["has_tax_rates"] = True

            chunked_docs.append(Document(text=chunk, metadata=metadata))

    return chunked_docs
