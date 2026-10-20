from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}



@dataclass
class DocumentPage:
    text: str
    source: str
    page: int | None = None
    metadata: dict | None = None
    
    
def _clean_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()

def load_pdf(path: Path) -> list[DocumentPage]:
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            pages.append(DocumentPage(text=text, source=str(path), page=page_no))
    return pages


def load_text(path: Path) -> list[DocumentPage]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = _clean_text(text)
    if not text:
        return []
    return [DocumentPage(text=text, source=str(path), page=None)]


def load_document(path: Path) -> list[DocumentPage]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".txt", ".md"}:
        return load_text(path)
    return []


def discover_documents(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return (
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
