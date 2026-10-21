from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

# from collections.abc import Iterable
import chromadb
from sentence_transformers import SentenceTransformer

from .chunker import Chunk

class RAGStore:
    def __init__(self, db_path:str | None= None, collection_name: str | None = None, embedding_model: str | None = None ):
        self.db_path = db_path or os.getenv("CHROMA_DIR", "artifacts/chroma")
        self.collection_name = collection_name or os.getenv("RAG_COLLECTION", "financial_docs")
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(self.embedding_model_name)
        
        
    @staticmethod
    def _id(chunk: Chunk) -> str:
        raw = f"{chunk.source}|{chunk.page}|{chunk.chunk_index}|{chunk.text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()    

    
    def add_chunks(self, chunks: Iterable[Chunk], batch_size: int = 64) -> int:
        chunks = list(chunks)
        if not chunks:
            return 0

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [c.text for c in batch]
            ids = [self._id(c) for c in batch]
            embeddings = self.embedder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=[c.metadata for c in batch],
            )
            total += len(batch)
        return total
    
    
    
    def search(self, query: str, k: int = 5) -> list[dict]:
        k = max(1, min(k, 20))
        q = self.embedder.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=q,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        rows: list[dict] = []
        for doc, meta, distance in zip(documents, metadatas, distances):
            # With cosine distance, lower is better. Convert it to a simple score.
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            rows.append({
                "text": doc,
                "metadata": meta or {},
                "score": round(score, 4),
            })
        return rows

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
