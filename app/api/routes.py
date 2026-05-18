import os
import tempfile
from pathlib import Path

from flask import Blueprint, request, jsonify

from services.rag_engine import query_tax_advisor
from services.vector_store import get_collection_info, ingest_documents
from services.document_processor import process_acts, process_file, chunk_documents

api_bp = Blueprint("api", __name__)


@api_bp.route("/health", methods=["GET"])
def health_check():
    try:
        info = get_collection_info()
        return jsonify({
            "status": "healthy",
            "vector_store": "connected" if info.get("exists") else "no_collection",
            "documents_count": info.get("documents_count", 0),
        })
    except Exception as e:
        return jsonify({
            "status": "degraded",
            "vector_store": f"error: {str(e)}",
            "documents_count": 0,
        })


@api_bp.route("/ingest", methods=["POST"])
def ingest_acts():
    try:
        documents = process_acts()
        docs_count = len(documents)
        chunked = chunk_documents(documents)
        lc_docs = [{"text": doc.text, "metadata": doc.metadata} for doc in chunked]
        total_chunks = ingest_documents(lc_docs)
        return jsonify({
            "status": "success",
            "documents_processed": docs_count,
            "total_chunks": total_chunks,
            "collection_name": "tax_acts",
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Ingestion failed: {str(e)}"}), 500


@api_bp.route("/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        # Rename temp file to use the original filename for metadata extraction
        named_path = tmp_path.parent / file.filename
        tmp_path.rename(named_path)
        tmp_path = named_path

        documents = process_file(tmp_path)
        chunked = chunk_documents(documents)
        lc_docs = [{"text": doc.text, "metadata": doc.metadata} for doc in chunked]
        total_chunks = ingest_documents(lc_docs)

        return jsonify({
            "status": "success",
            "filename": file.filename,
            "pages_processed": len(documents),
            "total_chunks": total_chunks,
        })
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500
    finally:
        if tmp_path and tmp_path.exists():
            os.unlink(tmp_path)


@api_bp.route("/query", methods=["POST"])
def query_tax():
    data = request.get_json()
    if not data or not data.get("query", "").strip():
        return jsonify({"error": "Query cannot be empty"}), 400
    try:
        result = query_tax_advisor(query=data["query"], top_k=data.get("top_k", 10))
        return jsonify({
            "query": result["query"],
            "summary": result["summary"],
            "detailed_analysis": result["detailed_analysis"],
            "sources": result["sources"],
            "tax_types_identified": result["tax_types_identified"],
        })
    except Exception as e:
        return jsonify({"error": f"Query processing failed: {str(e)}"}), 500
