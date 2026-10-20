from __future__ import annotations

import argparse
from pathlib import Path

from .chunker import chunk_pages
from .loaders import discover_documents, load_document
from .store import RAGStore


def index_directory(
    input_dir: str = "data/raw",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    clear: bool = False,
) -> tuple[int, int]:
    
    
    files = list(discover_documents(input_dir))
    if not files:
        raise RuntimeError(f"No PDF/TXT/MD files found under {input_dir}")

    store = RAGStore()
    if clear:
        store.clear()

    all_chunks = []
    for path in files:
        pages = load_document(path)
        chunks = chunk_pages(pages, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        print(f"Loaded {path} -> {len(chunks)} chunks")

    indexed = store.add_chunks(all_chunks)
    return len(files), indexed







def main() -> None:
    parser = argparse.ArgumentParser(description="Index financial PDF/TXT/MD documents into ChromaDB")
    parser.add_argument("--dir", default="data/raw")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    files, chunks = index_directory(
        args.dir,
        args.chunk_size,
        args.chunk_overlap,
        args.clear,
    )
    print(f"Indexed {chunks} chunks from {files} documents.")




if __name__ == "__main__":
    main()