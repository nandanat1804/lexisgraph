"""
Keyword search half of hybrid retrieval, using BM25 (rank_bm25).
Pure Python/CPU, trivially fast for laptop-scale document sets.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from .config import config
from .document_processor import Chunk

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Search:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]):
        self.chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self._save()
        print(f"[bm25] indexed {len(chunks)} chunks")

    def _save(self):
        Path(config.BM25_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(config.BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    def load(self) -> bool:
        p = Path(config.BM25_INDEX_PATH)
        if not p.exists():
            return False
        with open(p, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        return True

    def search(self, query: str, top_k: int = 10):
        if self.bm25 is None:
            if not self.load():
                return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in ranked_idx:
            if scores[idx] <= 0:
                continue
            c = self.chunks[idx]
            results.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_name": c.doc_name,
                    "page": c.page,
                    "text": c.text,
                    "score": float(scores[idx]),
                }
            )
        return results
