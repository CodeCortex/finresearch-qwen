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



