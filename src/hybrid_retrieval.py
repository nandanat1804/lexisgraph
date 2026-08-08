"""
Stage 3: Hybrid retrieval + RRF fusion
Combines dense (vector/semantic) search with sparse (BM25/keyword)
search using Reciprocal Rank Fusion, then reranks the fused list with
a cross-encoder for the final top-k passed to the LLM.

Why hybrid: semantic search is great for paraphrased/conceptual queries
("liability for late delivery") but weak on exact legal identifiers
("Section 12.3(b)", "Case No. 21-CV-4092"). BM25 nails exact terms but
misses paraphrases. RRF gets the best of both without needing to tune
a blend weight.
"""
from __future__ import annotations

from typing import Dict, List

from .bm25_search import BM25Search
from .config import config
from .reranker import rerank
from .vector_store import VectorStore


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]], k: int = 60
) -> List[Dict]:
    """
    Standard RRF: score(doc) = sum over lists of 1 / (k + rank_in_list)
    Ranks start at 1. Doc identity is by chunk_id.
    """
    fused_scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Dict] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            cid = doc["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank)
            doc_lookup[cid] = doc

    fused_sorted = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for cid, score in fused_sorted:
        doc = dict(doc_lookup[cid])
        doc["rrf_score"] = score
        results.append(doc)
    return results


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25: BM25Search):
        self.vector_store = vector_store
        self.bm25 = bm25

    def retrieve(self, query: str, final_k: int | None = None, use_rerank: bool = True) -> List[Dict]:
        final_k = final_k or config.TOP_K_RERANK

        vector_hits = self.vector_store.search(query, top_k=config.TOP_K_VECTOR)
        bm25_hits = self.bm25.search(query, top_k=config.TOP_K_BM25)

        fused = reciprocal_rank_fusion([vector_hits, bm25_hits], k=config.RRF_K)

        if not fused:
            return []

        if use_rerank:
            # Only rerank a reasonable candidate pool (cross-encoders are slower)
            candidate_pool = fused[: max(final_k * 3, 15)]
            return rerank(query, candidate_pool, top_k=final_k)

        return fused[:final_k]
