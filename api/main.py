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
    description="Financial research RAG + QLoRA inference API",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501").split(","),
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
    directory: str = "data/raw"
    chunk_size: int = Field(default=1200, ge=300, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    clear: bool = False
    
    
    
    
@app.get("/health")
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
    
@app.get("/v1/developer")
def developer():
    return {
        "developer": "Roshan Jaiswal (Codecortex)",
        "contact": {
        "linkedin": "https://www.linkedin.com/in/codecortex/",
        "instagram": "https://www.instagram.com/codecortexx/",
        "github": "https://github.com/CodeCortex"
        }
    }
    
    
