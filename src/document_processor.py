"""
Stage 1: Documents -> preprocessing
Extracts text from PDF/TXT legal documents and splits into overlapping
chunks suitable for embedding. Pure-CPU, no external services.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import config


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    text: str
    metadata: dict = field(default_factory=dict)


def _clean_text(text: str) -> str:
    """Normalize whitespace, strip stray page-break artifacts."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"-\s+", "-", text)  # de-hyphenate line-wrapped words
    return text.strip()


def extract_pdf(path: Path) -> List[tuple[int, str]]:
    """Returns list of (page_number, raw_text)."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        if raw.strip():
            pages.append((i, raw))
    return pages


def extract_txt(path: Path) -> List[tuple[int, str]]:
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def load_document_pages(path: Path) -> List[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    elif suffix in (".txt", ".md"):
        return extract_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt")


def chunk_document(path: Path, doc_id: str | None = None) -> List[Chunk]:
    """
    Full pipeline for a single document: extract -> clean -> chunk.
    Keeps page-level provenance for citation in answers.
    """
    doc_id = doc_id or str(uuid.uuid4())[:8]
    doc_name = path.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Chunk] = []
    for page_num, raw_text in load_document_pages(path):
        cleaned = _clean_text(raw_text)
        if not cleaned:
            continue
        for piece in splitter.split_text(cleaned):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{uuid.uuid4().hex[:8]}",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page=page_num,
                    text=piece,
                    metadata={"source": doc_name, "page": page_num},
                )
            )
    return chunks


def chunk_directory(directory: Path) -> List[Chunk]:
    """Process every .pdf/.txt/.md file in a directory."""
    all_chunks: List[Chunk] = []
    files = sorted(
        [p for p in directory.glob("*") if p.suffix.lower() in (".pdf", ".txt", ".md")]
    )
    if not files:
        raise FileNotFoundError(f"No .pdf/.txt/.md files found in {directory}")
    for path in files:
        doc_id = uuid.uuid4().hex[:8]
        chunks = chunk_document(path, doc_id=doc_id)
        print(f"  [doc_processor] {path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    return all_chunks
