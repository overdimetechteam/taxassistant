"""
Qdrant vector store service.
Manages collection creation, document ingestion, and similarity search
with optimized payload schema and indexing for tax act retrieval.
Uses Qdrant Dedicated Cloud Cluster.
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

from core.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_CLUSTER_ENDPOINT,
    QDRANT_COLLECTION_NAME,
    QDRANT_PREFER_GRPC,
    QDRANT_BATCH_SIZE,
)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_CLUSTER_ENDPOINT,
        api_key=QDRANT_API_KEY,
        prefer_grpc=QDRANT_PREFER_GRPC,
    )


def _create_payload_indexes(client: QdrantClient):
    """Create payload indexes — skips silently if index already exists."""
    index_configs = [
        ("metadata.tax_type_code", PayloadSchemaType.KEYWORD),
        ("metadata.year",          PayloadSchemaType.INTEGER),
        ("metadata.act_number",    PayloadSchemaType.KEYWORD),
        ("metadata.tax_type",      PayloadSchemaType.KEYWORD),
        ("metadata.has_tax_rates", PayloadSchemaType.BOOL),
    ]
    for field, schema in index_configs:
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass  # index already exists

    try:
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
    except Exception:
        pass


def _ensure_collection(client: QdrantClient, vector_size: int = 3072):
    """Create the collection and indexes if they don't exist yet."""
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        _create_payload_indexes(client)


def get_vector_store() -> QdrantVectorStore:
    embeddings = get_embeddings()
    return QdrantVectorStore.from_existing_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings,
        url=QDRANT_CLUSTER_ENDPOINT,
        api_key=QDRANT_API_KEY,
        prefer_grpc=QDRANT_PREFER_GRPC,
    )


def ingest_documents(documents: list[dict], batch_size: int | None = None) -> int:
    """
    Append document chunks to the existing Qdrant collection.
    Creates the collection if it does not exist yet.
    Never wipes existing data.

    Args:
        documents:  List of dicts with 'text' and 'metadata' keys.
        batch_size: Chunks per upload batch (defaults to QDRANT_BATCH_SIZE).

    Returns:
        Number of chunks ingested in this call.
    """
    if not documents:
        return 0

    batch_size = batch_size or QDRANT_BATCH_SIZE
    embeddings = get_embeddings()

    from langchain_core.documents import Document as LCDocument

    lc_docs = [
        LCDocument(page_content=doc["text"], metadata=doc["metadata"])
        for doc in documents
    ]

    # Ensure collection exists before adding
    client = get_qdrant_client()
    try:
        _ensure_collection(client)
    finally:
        client.close()

    # Add all batches to the existing collection
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings,
        url=QDRANT_CLUSTER_ENDPOINT,
        api_key=QDRANT_API_KEY,
        prefer_grpc=QDRANT_PREFER_GRPC,
    )

    total_batches = (len(lc_docs) + batch_size - 1) // batch_size
    for i in range(0, len(lc_docs), batch_size):
        batch = lc_docs[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"      Batch {i // batch_size + 1}/{total_batches} uploaded ({len(batch)} docs)")

    return len(lc_docs)


def get_collection_info() -> dict:
    client = get_qdrant_client()
    try:
        existing = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in existing:
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
