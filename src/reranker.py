"""
Reranking stage: takes the fused candidate list from hybrid retrieval
and re-scores each (query, chunk) pair with a small ONNX cross-encoder
(via `fastembed`, not torch). Cross-encoders are much more accurate than
bi-encoder cosine similarity because they attend over the query and
document jointly - but they're slower, which is why we only run them on
the top ~15-20 candidates instead of the whole corpus.

Uses ONNX runtime instead of torch/sentence-transformers for the same
reason as embeddings.py: a far smaller memory footprint, which matters
on free-tier hosts with limited RAM.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Dict

from fastembed.rerank.cross_encoder import TextCrossEncoder

from .config import config


@lru_cache(maxsize=1)
def get_reranker() -> TextCrossEncoder:
    print(f"[reranker] loading {config.RERANKER_MODEL} (ONNX/CPU, lightweight)...")
    return TextCrossEncoder(model_name=config.RERANKER_MODEL)


def rerank(query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    if not candidates:
        return []
    model = get_reranker()
    documents = [c["text"] for c in candidates]
    scores = list(model.rerank(query, documents))
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
