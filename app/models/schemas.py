from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The tax scenario or question to analyze")
    top_k: int = Field(default=10, description="Number of relevant chunks to retrieve")


class SourceDocument(BaseModel):
    act_name: str
    tax_type: str
    year: int | None
    act_number: str
    section: str
    page: int | None
    content_preview: str


class QueryResponse(BaseModel):
    query: str
    summary: str
    detailed_analysis: str
    sources: list[SourceDocument]
    tax_types_identified: list[str]


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    total_chunks: int
    collection_name: str


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    documents_count: int
