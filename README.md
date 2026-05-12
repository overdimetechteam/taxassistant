# Sri Lanka Tax Advisor - RAG Application

A Retrieval-Augmented Generation (RAG) application that acts as a Sri Lankan tax advisor. It ingests legislative tax acts, stores them in a Qdrant vector database with rich metadata, and uses Gemini 2.5 Flash to provide comprehensive tax analysis for any given business scenario.

## Tech Stack

| Component              | Technology                        |
|------------------------|-----------------------------------|
| **LLM**                | Google Gemini 2.5 Flash           |
| **Embeddings**         | Google text-embedding-004         |
| **Vector Database**    | Qdrant (local file-based storage) |
| **Document Processing**| LlamaIndex + PyMuPDF              |
| **RAG Orchestration**  | LangChain                         |
| **API Framework**      | FastAPI                           |
| **Language**           | Python 3.11+                      |

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Tax Act PDFs│────►│  LlamaIndex      │────►│  Qdrant         │
│  (/Acts)     │     │  Document Proc.  │     │  Vector Store   │
└──────────────┘     │  - PDF parsing   │     │  - Vectors      │
                     │  - Metadata ext. │     │  - Metadata     │
                     │  - Chunking      │     │  - Payload idx  │
                     └──────────────────┘     └────────┬────────┘
                                                       │
┌──────────────┐     ┌──────────────────┐              │
│  User Query  │────►│  LangChain RAG   │◄─────────────┘
│  (FastAPI)   │     │  - Retriever     │
└──────────────┘     │  - Gemini 2.5    │
                     │  - Tax Advisor   │
       ▲             │    System Prompt │
       │             └────────┬─────────┘
       │                      │
       └──────────────────────┘
         Structured JSON Response
         (answer + sources + tax types)
```

## Vector DB Schema Design

The Qdrant collection uses a carefully designed schema optimized for tax legislation retrieval:

### Vector Configuration
- **Embedding Model**: Google `text-embedding-004` (768 dimensions)
- **Distance Metric**: Cosine similarity
- **Storage**: Local file-based (no server needed)

### Metadata Payload Schema

Each document chunk is stored with the following metadata fields:

| Field                 | Type      | Indexed | Purpose                                           |
|-----------------------|-----------|---------|---------------------------------------------------|
| `source_file`         | `string`  | No      | Original PDF filename                             |
| `tax_type`            | `keyword` | **Yes** | Human-readable tax type (e.g., "Value Added Tax") |
| `tax_type_code`       | `keyword` | **Yes** | Short code (VAT, NBT, BNG, IRA, SSCL)            |
| `act_name`            | `string`  | No      | Full act name derived from filename               |
| `act_number`          | `keyword` | **Yes** | Legislative act number                            |
| `year`                | `integer` | **Yes** | Most recent year associated with the act          |
| `is_amendment`        | `bool`    | No      | Whether this is an amendment act                  |
| `is_consolidation`    | `bool`    | No      | Whether this is a consolidated act                |
| `section`             | `text`    | **Yes** | Detected section/part/schedule reference          |
| `page_number`         | `integer` | No      | Page number in source PDF                         |
| `tax_rates_mentioned` | `string`  | No      | Comma-separated tax rates found in chunk          |
| `has_tax_rates`       | `bool`    | **Yes** | Quick flag for chunks containing rate info        |
| `language`            | `string`  | No      | Document language                                 |

### Why This Schema?

1. **Keyword indexes on `tax_type_code` and `act_number`**: Enables fast pre-filtering. When a user asks about VAT, the system can narrow the search space before computing vector similarity.

2. **Integer index on `year`**: Supports temporal queries like "latest VAT rate" by filtering to the most recent acts first.

3. **Bool index on `has_tax_rates`**: Quickly locates chunks that contain actual rate information, which is critical for rate lookups.

4. **Full-text index on `section`**: Allows section-level search within acts (e.g., "Section 2 of VAT Act").

5. **Rich metadata preservation**: Every chunk carries its full lineage (source file, act, page, type) enabling precise source attribution in responses.

### Retrieval Strategy

The system retrieves **15 chunks by default** using cosine similarity, then passes them as context to Gemini 2.5 Flash. This higher-than-usual `k` value ensures comprehensive coverage across multiple tax types that may apply to a single scenario.

## Tax Acts Covered

The system processes the following Sri Lankan tax legislation from the `/Acts` folder:

| Tax Type                          | Code | Acts Included                                      |
|-----------------------------------|------|-----------------------------------------------------|
| Value Added Tax                   | VAT  | VAT Act No. 14 of 2002 (Consolidation 2025), Amendment Act No. 16 of 2024 |
| Nation Building Tax               | NBT  | NBT Acts: No. 9 of 2009 (Cons. 2013), No. 10/2014, No. 12/2015, No. 13/2017, No. 22/2016, No. 20/2018, No. 20/2019, No. 3/2020 |
| Betting and Gaming Levy           | BNG  | BNG Acts: No. 40 of 1988 (Cons. 2013), No. 14/2015, No. 11/2023, No. 25/2025 |
| Inland Revenue (Income Tax)       | IRA  | IRA Consolidated Act - 2025 Changes                |
| Social Security Contribution Levy | SSCL | SSCL Act No. 25 of 2022                            |
| Other                             | -    | Act 24-2025, Act 6379, Act 6380                    |

## Setup & Installation

### Prerequisites
- Python 3.11 or higher
- Google Cloud account with Gemini API access

### 1. Clone and Setup

```bash
cd TaxAssistant

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example env file
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux

# Edit .env and add your Google API key
# GOOGLE_API_KEY=your_key_here
```

### 3. Ingest Tax Acts

```bash
python ingest.py
```

This will:
- Parse all 19 PDF files from the `/Acts` folder
- Extract metadata (tax type, act number, year, amendments, etc.)
- Chunk documents using sentence-aware splitting (1024 tokens, 200 overlap)
- Generate embeddings using Google text-embedding-004
- Store vectors + metadata in Qdrant (local file storage in `./qdrant_data`)
- Create payload indexes for fast filtered retrieval

### 4. Start the API Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

## API Endpoints

### `GET /` - Root
Returns application info and available endpoints.

### `GET /api/v1/health` - Health Check
Check API status and vector store connectivity.

**Response:**
```json
{
  "status": "healthy",
  "vector_store": "connected",
  "documents_count": 1523
}
```

### `POST /api/v1/ingest` - Ingest Documents
Process and ingest all tax acts from the Acts folder. Run this when you add new acts.

**Response:**
```json
{
  "status": "success",
  "documents_processed": 450,
  "total_chunks": 1523,
  "collection_name": "tax_acts"
}
```

### `POST /api/v1/query` - Query Tax Advisor
Submit a tax scenario for analysis.

**Request:**
```json
{
  "query": "I am importing leather shoes from Italy to sell in Sri Lanka. What are all the tax implications?",
  "top_k": 15
}
```

**Response:**
```json
{
  "query": "I am importing leather shoes from Italy...",
  "answer": "Based on the Sri Lankan tax legislation, importing leather shoes involves the following tax implications:\n\n1. **Value Added Tax (VAT)**: ...\n2. **Nation Building Tax (NBT)**: ...\n3. **Social Security Contribution Levy (SSCL)**: ...",
  "sources": [
    {
      "act_name": "VAT Act No 14 E 2002 (Consolidation 2025)",
      "tax_type": "Value Added Tax",
      "year": 2025,
      "act_number": "14",
      "section": "Section 2",
      "page": 5,
      "content_preview": "..."
    }
  ],
  "tax_types_identified": ["Value Added Tax", "Nation Building Tax", "Social Security Contribution Levy"]
}
```

## Example Queries

```
"What VAT rate applies to importing electronic goods?"

"I run a restaurant in Colombo. What taxes do I need to pay?"

"What are the tax implications for selling leather shoes in Sri Lanka?"

"I want to start an online betting platform. What are the applicable levies?"

"What income tax rates apply to a company with annual revenue of 500 million LKR?"

"Are there any VAT exemptions for essential food items?"

"What is the SSCL rate for a manufacturing company?"

"I am exporting tea. What tax benefits or exemptions apply?"
```

## Project Structure

```
TaxAssistant/
├── Acts/                          # Source tax act PDFs (19 files)
│   ├── VAT_Act_No_14[E]_2002_(Consolidation_2025).pdf
│   ├── IRA_Cons_Act_-_2025_Changes.pdf
│   ├── NBT_Act_No_9(E)2009_(Consolidation_2013).pdf
│   └── ...
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # API endpoint definitions
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py              # Configuration and environment variables
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic request/response models
│   └── services/
│       ├── __init__.py
│       ├── document_processor.py  # LlamaIndex PDF parsing & chunking
│       ├── vector_store.py        # Qdrant vector store management
│       └── rag_engine.py          # LangChain RAG query engine
├── qdrant_data/                   # Local Qdrant storage (created after ingestion)
├── ingest.py                      # Standalone ingestion script
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
└── README.md                      # This file
```

## How the RAG Pipeline Works

### 1. Document Ingestion (LlamaIndex)
- **PDF Parsing**: PyMuPDF extracts text from each page of every tax act PDF
- **Metadata Extraction**: Filenames are parsed to extract tax type, act number, year, amendment status using regex patterns
- **Section Detection**: Each chunk is scanned for section/part/schedule references (e.g., "Section 25A", "PART III", "Schedule II")
- **Rate Extraction**: Tax rate percentages are detected and stored as metadata for quick rate lookups
- **Sentence-Aware Chunking**: Documents are split into 1024-token chunks with 200-token overlap, respecting sentence boundaries to preserve semantic coherence

### 2. Vector Storage (Qdrant)
- Chunks are embedded using Google's `text-embedding-004` model (768 dimensions)
- Stored in Qdrant with cosine similarity distance metric
- Six payload indexes are created for fast filtered retrieval (see Schema section)
- Local file-based storage requires zero infrastructure

### 3. Query Processing (LangChain + Gemini)
- User query is embedded and used for cosine similarity search against stored vectors
- Top 15 matching chunks are retrieved with full metadata
- A tax-advisor system prompt instructs Gemini to:
  - Identify ALL applicable tax types
  - Be vigilant about product classifications (e.g., shoes → leather products)
  - Quote exact rates and section references
  - Prefer the latest/amended rates
  - Flag exemptions and thresholds
- Response includes the analysis, source references, and identified tax types

## Configuration

All configuration is managed through environment variables (`.env` file):

| Variable                | Default                 | Description                          |
|-------------------------|-------------------------|--------------------------------------|
| `GOOGLE_API_KEY`        | *(required)*            | Google Gemini API key                |
| `GEMINI_MODEL`          | `gemini-2.5-flash`      | Gemini model for response generation |
| `EMBEDDING_MODEL`       | `models/text-embedding-004` | Embedding model                  |
| `QDRANT_LOCAL_PATH`     | `./qdrant_data`         | Local Qdrant storage path            |
| `QDRANT_COLLECTION_NAME`| `tax_acts`              | Qdrant collection name               |
| `CHUNK_SIZE`            | `1024`                  | Token count per chunk                |
| `CHUNK_OVERLAP`         | `200`                   | Token overlap between chunks         |
| `ACTS_FOLDER`           | `./Acts`                | Path to tax act PDFs                 |

## Key Design Decisions

1. **Qdrant over other vector DBs**: Chosen for its payload indexing capabilities which enable hybrid search (vector + metadata filtering) critical for tax legislation where filtering by tax type, year, or act number dramatically improves relevance.

2. **LlamaIndex for ingestion, LangChain for RAG**: LlamaIndex excels at document parsing and node/chunk management. LangChain provides a more flexible chain composition for the RAG query pipeline with Gemini integration.

3. **High retrieval count (k=15)**: Tax scenarios often span multiple tax types (VAT + NBT + SSCL + Income Tax). A higher k ensures the LLM has access to all relevant legislation rather than just the most similar chunks.

4. **Rich metadata schema**: Instead of flat text storage, each chunk carries its full provenance (act, section, page, year, rates). This enables both precise source attribution and potential metadata-filtered retrieval.

5. **Low LLM temperature (0.1)**: Tax advice requires factual accuracy, not creativity. A low temperature ensures the model stays close to the provided legislative context.

6. **Local Qdrant storage**: No separate Qdrant server needed. The `qdrant_data/` folder stores everything locally, making deployment simpler for development and small-scale use.
