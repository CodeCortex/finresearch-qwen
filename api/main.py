from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.generate import generate
from src.rag.ingest import index_directory
from src.rag.store import RAGStore

app = FastAPI(
    title="FinResearch-Qwen API",
    version="2.0.0",
    description=(
        "Financial research API powered by "
        "Qwen3-4B + QLoRA + RAG"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)



cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8501"
).split(",")

cors_origins = [
    origin.strip()
    for origin in cors_origins
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_rag() -> RAGStore:
    return RAGStore()


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    use_rag: bool = True
    top_k: int = Field(default=5, ge=1, le=20)
    

class IndexRequest(BaseModel):
    directory: str = "data/raw/company_reports"
    chunk_size: int = Field(default=1200, ge=300, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    clear: bool = False
    
    
    
    
@app.get("/health", tags=["System"])
def health():
    try:
        count = get_rag().count()
    except Exception:
        count = 0
    return {
        "status": "ok",
        "service": "finresearch-qwen",
        "indexed_chunks": count,
    }


    
@app.get("/v1/developer", tags=["System"])
def developer():
    return {
        "developer": "Roshan Jaiswal (Codecortex)",
        "contact": {
        "linkedin": "https://www.linkedin.com/in/codecortex/",
        "instagram": "https://www.instagram.com/codecortexx/",
        "github": "https://github.com/CodeCortex"
        }
    }
    
    
  
@app.post("/v1/research", tags=["Research"])
def research(request: ResearchRequest):
    try:
        results = get_rag().search(request.question, request.top_k) if request.use_rag else []
        context = "\n\n---\n\n".join(
            f"Source: {r['metadata'].get('source', 'unknown')}\n"
            f"Page: {r['metadata'].get('page', 'N/A')}\n"
            f"Evidence:\n{r['text']}"
            for r in results
        )
        answer = generate(request.question, context)
        sources = [
            {
                "source": r["metadata"].get("source", "unknown"),
                "file_name": r["metadata"].get("file_name", "unknown"),
                "page": r["metadata"].get("page", None),
                "score": r["score"],
                "snippet": r["text"][:500],
            }
            for r in results
        ]
        return {"answer": answer, "sources": sources}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
  




@app.post("/v1/index", tags=["RAG"])
def index(request: IndexRequest):
    try:
        files, chunks = index_directory(
            input_dir=request.directory,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            clear=request.clear,
        )
        # Re-create cached RAG store
        get_rag.cache_clear()
        return {"status": "ok", "documents": files, "chunks": chunks}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
