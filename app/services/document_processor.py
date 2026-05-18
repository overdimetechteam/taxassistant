"""
Document processing pipeline using LlamaIndex.
Handles PDF parsing, metadata extraction, and intelligent chunking
for Sri Lankan tax acts.
"""

import re
from pathlib import Path

import fitz  # pymupdf
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from core.config import ACTS_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, TAX_TYPE_MAP


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

    # Detect tax type from prefix
    for code, full_name in TAX_TYPE_MAP.items():
        if filename.upper().startswith(code):
            metadata["tax_type"] = full_name
            metadata["tax_type_code"] = code
            break

    # Extract act number
    act_no_match = re.search(r"No[._\s]*(\d+)", filename)
    if act_no_match:
        metadata["act_number"] = act_no_match.group(1)

    # Extract years - take the latest year found as the primary year
    years = re.findall(r"((?:19|20)\d{2})", filename)
    if years:
        metadata["year"] = max(int(y) for y in years)

    # Check if amendment
    if "Amd" in filename or "Amendment" in filename:
        metadata["is_amendment"] = True

    # Check if consolidation
    if "Consolidation" in filename or "Cons" in filename:
        metadata["is_consolidation"] = True

    # Detect language
    if "[E]" in filename or "_E." in filename or "_E_" in filename:
        metadata["language"] = "English"

    return metadata


def _detect_section_from_text(text: str) -> str:
    """Try to identify the section/part number from chunk text."""
    # Match patterns like "Section 2", "PART II", "Schedule I", "Article 5"
    patterns = [
        r"(?:Section|SECTION)\s+(\d+[A-Za-z]*)",
        r"(?:PART|Part)\s+([IVXLCDM]+|\d+)",
        r"(?:Schedule|SCHEDULE)\s+([IVXLCDM]+|\d+)",
        r"(?:Article|ARTICLE)\s+(\d+)",
        r"(?:Chapter|CHAPTER)\s+([IVXLCDM]+|\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:500])  # Only look at start of chunk
        if match:
            return match.group(0)
    return ""


def _extract_tax_rates_from_text(text: str) -> list[str]:
    """Extract any tax rate percentages mentioned in the text."""
    rates = re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|per\s*cent|percent)", text, re.IGNORECASE)
    return list(set(rates))


def parse_pdf(pdf_path: Path) -> list[dict]:
    """
    Parse a PDF file using PyMuPDF and return pages with text and metadata.
    """
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


def process_file(pdf_path: Path) -> list[Document]:
    """Process a single PDF file into LlamaIndex Documents with rich metadata."""
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


def process_acts(acts_folder: Path | None = None) -> list[Document]:
    """
    Process all PDF acts from the Acts folder into LlamaIndex Documents
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

            doc = Document(text=text, metadata=metadata)
            all_documents.append(doc)

    return all_documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into optimally-sized chunks using sentence-aware splitting.
    Preserves metadata across chunks.
    """
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        paragraph_separator="\n\n",
    )

    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

    chunked_docs = []
    for node in nodes:
        # Re-detect section for each chunk since splitting may change context
        section = _detect_section_from_text(node.text)
        tax_rates = _extract_tax_rates_from_text(node.text)

        metadata = dict(node.metadata)
        if section:
            metadata["section"] = section
        if tax_rates:
            metadata["tax_rates_mentioned"] = ", ".join(tax_rates)
            metadata["has_tax_rates"] = True

        chunked_docs.append(Document(text=node.text, metadata=metadata))

    return chunked_docs
