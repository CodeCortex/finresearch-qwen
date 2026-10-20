from __future__ import annotations

from dataclasses import dataclass

from .loaders import DocumentPage


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None
    chunk_index: int
    metadata: dict


def split_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(paragraph):
                    break
                start = end - chunk_overlap
            continue

        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{tail}\n{paragraph}".strip() if tail else paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def chunk_pages(pages: list[DocumentPage], chunk_size: int = 1200, chunk_overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        pieces = split_text(page.text, chunk_size, chunk_overlap)
        for idx, piece in enumerate(pieces):
            metadata = {
                "source": page.source,
                "file_name": page.source.split("/")[-1],
                "page": page.page if page.page is not None else -1,
                "chunk_index": idx,
            }
            chunks.append(
                Chunk(
                    text=piece,
                    source=page.source,
                    page=page.page,
                    chunk_index=idx,
                    metadata=metadata,
                )
            )
    return chunks
