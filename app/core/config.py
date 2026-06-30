import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ACTS_FOLDER = Path(os.getenv("ACTS_FOLDER", str(BASE_DIR / "Acts")))

# Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Qdrant Dedicated Cloud Cluster
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "tax_acts")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT", "")
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "500"))

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Tax type mappings derived from filename prefixes
TAX_TYPE_MAP = {
    "VAT": "Value Added Tax",
    "NBT": "Nation Building Tax",
    "BNG": "Betting and Gaming Levy",
    "IRA": "Inland Revenue Act",
    "SSCL": "Social Security Contribution Levy",
}
