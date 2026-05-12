"""
FastAPI route definitions for the Tax Advisor RAG API.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    HealthResponse,
    SourceDocument,
)
from app.services.rag_engine import query_tax_advisor
from app.services.vector_store import get_collection_info, ingest_documents
from app.services.document_processor import process_acts, chunk_documents

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and vector store status."""
    try:
        info = get_collection_info()
        return HealthResponse(
            status="healthy",
            vector_store="connected" if info.get("exists") else "no_collection",
            documents_count=info.get("documents_count", 0),
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            vector_store=f"error: {str(e)}",
            documents_count=0,
        )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_acts():
    """
    Process and ingest all tax acts from the Acts folder into Qdrant.
    This parses PDFs, extracts metadata, chunks documents, and stores
    them in the vector database.
    """
    try:
        # Step 1: Parse PDFs and extract metadata using LlamaIndex
        documents = process_acts()
        docs_count = len(documents)

        # Step 2: Chunk documents with sentence-aware splitting
        chunked = chunk_documents(documents)

        # Step 3: Prepare for LangChain/Qdrant ingestion
        lc_docs = [
            {"text": doc.text, "metadata": doc.metadata}
            for doc in chunked
        ]

        # Step 4: Ingest into Qdrant
        total_chunks = ingest_documents(lc_docs)

        return IngestResponse(
            status="success",
            documents_processed=docs_count,
            total_chunks=total_chunks,
            collection_name="tax_acts",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_tax(request: QueryRequest):
    """
    Query the tax advisor with a scenario or question.
    Returns comprehensive tax implications with source references.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = query_tax_advisor(query=request.query, top_k=request.top_k)

        sources = [
            SourceDocument(**src) for src in result["sources"]
        ]

        return QueryResponse(
            query=result["query"],
            summary=result["summary"],
            detailed_analysis=result["detailed_analysis"],
            sources=sources,
            tax_types_identified=result["tax_types_identified"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}",
        )
