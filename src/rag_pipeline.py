"""
Orchestrates the full pipeline:
Documents -> preprocessing -> retrieval -> hybrid search -> reranking
-> knowledge graph enrichment -> LLM -> answer
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

from .bm25_search import BM25Search
from .document_processor import chunk_directory
from .hybrid_retrieval import HybridRetriever
from .knowledge_graph import KnowledgeGraph
from .llm_client import generate_answer
from .vector_store import VectorStore


class LexisGraphPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25 = BM25Search()
        self.kg = KnowledgeGraph()
        self._loaded = False

    # ---------------- Ingestion ----------------
    def ingest(self, docs_dir: str):
        docs_path = Path(docs_dir)
        print(f"[pipeline] ingesting documents from {docs_path}")
        chunks = chunk_directory(docs_path)

        print("[pipeline] building vector index...")
        self.vector_store.index_chunks(chunks)

        print("[pipeline] building BM25 index...")
        self.bm25.build(chunks)

        print("[pipeline] building knowledge graph...")
        self.kg.build(chunks)

        self._loaded = True
        print(f"[pipeline] ingestion complete: {len(chunks)} chunks indexed")
        return len(chunks)

    def _ensure_loaded(self):
        if self._loaded:
            return
        self.bm25.load()
        self.kg.load()
        self._loaded = True

    # ---------------- Query ----------------
    def query(self, question: str, top_k: int = 5, use_kg: bool = True) -> Dict:
        self._ensure_loaded()
        t0 = time.time()

        retriever = HybridRetriever(self.vector_store, self.bm25)
        results = retriever.retrieve(question, final_k=top_k)

        kg_extra_ids = []
        if use_kg:
            kg_extra_ids = self.kg.related_chunks_for_entity(question)

        answer = generate_answer(question, results)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"doc_name": r["doc_name"], "page": r["page"],
                 "rerank_score": r.get("rerank_score"), "text": r["text"][:250]}
                for r in results
            ],
            "kg_related_chunk_count": len(kg_extra_ids),
            "latency_seconds": round(time.time() - t0, 2),
        }

    def status(self) -> Dict:
        self._ensure_loaded()
        return {
            "vector_chunks": self.vector_store.count(),
            "bm25_chunks": len(self.bm25.chunks),
            "knowledge_graph": self.kg.stats(),
        }
