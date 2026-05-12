"""
Standalone script to ingest tax act PDFs into the Qdrant vector store.
Run this before starting the API server.

Usage:
    python ingest.py
"""

import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import ACTS_FOLDER, QDRANT_COLLECTION_NAME
from app.services.document_processor import process_acts, chunk_documents
from app.services.vector_store import ingest_documents


def main():
    print("=" * 60)
    print("  Tax Acts Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Parse PDFs
    print(f"\n[1/3] Parsing PDFs from: {ACTS_FOLDER}")
    start = time.time()
    documents = process_acts()
    elapsed = time.time() - start
    print(f"      Parsed {len(documents)} pages from PDF files ({elapsed:.1f}s)")

    # Show summary of parsed documents
    tax_types = {}
    for doc in documents:
        tt = doc.metadata.get("tax_type", "Unknown")
        tax_types[tt] = tax_types.get(tt, 0) + 1
    print("\n      Documents by tax type:")
    for tt, count in sorted(tax_types.items()):
        print(f"        - {tt}: {count} pages")

    # Step 2: Chunk documents
    print(f"\n[2/3] Chunking documents...")
    start = time.time()
    chunked = chunk_documents(documents)
    elapsed = time.time() - start
    print(f"      Created {len(chunked)} chunks ({elapsed:.1f}s)")

    # Step 3: Ingest into Qdrant
    print(f"\n[3/3] Ingesting into Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
    start = time.time()

    lc_docs = [
        {"text": doc.text, "metadata": doc.metadata}
        for doc in chunked
    ]

    total = ingest_documents(lc_docs)
    elapsed = time.time() - start
    print(f"      Ingested {total} chunks into vector store ({elapsed:.1f}s)")

    print("\n" + "=" * 60)
    print("  Ingestion complete!")
    print(f"  Collection: {QDRANT_COLLECTION_NAME}")
    print(f"  Total vectors: {total}")
    print("=" * 60)
    print("\nYou can now start the API server:")
    print("  uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
