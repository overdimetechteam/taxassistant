"""
Tax Advisor RAG Application
Built with FastAPI + LangChain + LlamaIndex + Qdrant + Gemini 2.5 Flash
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Sri Lanka Tax Advisor API",
    description=(
        "RAG-powered tax advisory system that analyzes Sri Lankan tax acts "
        "to identify all applicable tax implications, rates, and exemptions "
        "for any given business scenario."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
