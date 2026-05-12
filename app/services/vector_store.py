"""
Qdrant vector store service.
Manages collection creation, document ingestion, and similarity search
with optimized payload schema and indexing for tax act retrieval.
Uses Qdrant Cloud.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
)
from langchain_qdrant import QdrantVectorStore

from app.core.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_CLUSTER_ENDPOINT,
    QDRANT_COLLECTION_NAME,
)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Get the Google Generative AI embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client connected to Qdrant Cloud with extended timeout."""
    return QdrantClient(
        url=QDRANT_CLUSTER_ENDPOINT,
        api_key=QDRANT_API_KEY,
        timeout=720,
    )


def _create_payload_indexes(client: QdrantClient):
    """Create all payload indexes on the collection for fast filtered search."""
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="metadata.tax_type_code",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="metadata.year",
        field_schema=PayloadSchemaType.INTEGER,
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="metadata.act_number",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="metadata.tax_type",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="metadata.has_tax_rates",
        field_schema=PayloadSchemaType.BOOL,
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="metadata.section",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WORD,
            min_token_len=1,
            max_token_len=20,
            lowercase=True,
        ),
    )


def create_collection_with_schema(client: QdrantClient, vector_size: int = 3072):
    """
    Create Qdrant collection with optimized schema for tax act retrieval.
    """
    collections = client.get_collections().collections
    existing_names = [c.name for c in collections]

    if QDRANT_COLLECTION_NAME in existing_names:
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    _create_payload_indexes(client)


def get_vector_store() -> QdrantVectorStore:
    """
    Get LangChain-compatible Qdrant vector store instance.
    Connects to Qdrant Cloud.
    """
    embeddings = get_embeddings()

    return QdrantVectorStore.from_existing_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings,
        url=QDRANT_CLUSTER_ENDPOINT,
        api_key=QDRANT_API_KEY,
    )


def ingest_documents(documents: list[dict], batch_size: int = 300) -> int:
    """
    Ingest processed document chunks into Qdrant Cloud in batches.

    Args:
        documents: List of dicts with 'text' and 'metadata' keys.
        batch_size: Number of documents per upload batch.

    Returns:
        Number of documents ingested.
    """
    embeddings = get_embeddings()

    from langchain_core.documents import Document as LCDocument

    lc_docs = [
        LCDocument(page_content=doc["text"], metadata=doc["metadata"])
        for doc in documents
    ]

    # First batch creates the collection (force_recreate)
    first_batch = lc_docs[:batch_size]
    QdrantVectorStore.from_documents(
        documents=first_batch,
        embedding=embeddings,
        url=QDRANT_CLUSTER_ENDPOINT,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME,
        force_recreate=True,
        timeout=720,
    )
    print(f"      Batch 1/{(len(lc_docs) + batch_size - 1) // batch_size} uploaded ({len(first_batch)} docs)")

    # Remaining batches add to existing collection
    if len(lc_docs) > batch_size:
        vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=embeddings,
            url=QDRANT_CLUSTER_ENDPOINT,
            api_key=QDRANT_API_KEY,
        )
        for i in range(batch_size, len(lc_docs), batch_size):
            batch = lc_docs[i:i + batch_size]
            vector_store.add_documents(batch)
            batch_num = (i // batch_size) + 1
            total_batches = (len(lc_docs) + batch_size - 1) // batch_size
            print(f"      Batch {batch_num}/{total_batches} uploaded ({len(batch)} docs)")

    # Apply payload indexes
    client = get_qdrant_client()
    _create_payload_indexes(client)
    client.close()

    return len(lc_docs)


def get_collection_info() -> dict:
    """Get information about the Qdrant Cloud collection."""
    client = get_qdrant_client()
    try:
        collections = client.get_collections().collections
        existing_names = [c.name for c in collections]

        if QDRANT_COLLECTION_NAME not in existing_names:
            return {"exists": False, "documents_count": 0}

        info = client.get_collection(QDRANT_COLLECTION_NAME)
        return {
            "exists": True,
            "documents_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.value,
        }
    finally:
        client.close()
