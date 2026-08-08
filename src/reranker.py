"""
Reranking stage: takes the fused candidate list from hybrid retrieval
and re-scores each (query, chunk) pair with a small cross-encoder.
Cross-encoders are much more accurate than bi-encoder cosine similarity
because they attend over the query and document jointly - but they're
slower, which is why we only run them on the top ~15-20 candidates
instead of the whole corpus.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Dict

from sentence_transformers import CrossEncoder

from .config import config


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    print(f"[reranker] loading {config.RERANKER_MODEL} (CPU)...")
    return CrossEncoder(config.RERANKER_MODEL, device="cpu")


def rerank(query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    if not candidates:
        return []
    model = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
