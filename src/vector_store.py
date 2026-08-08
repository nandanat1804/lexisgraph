"""
Vector store using Qdrant in EMBEDDED/LOCAL mode.
No Docker, no server, no network required — Qdrant runs as an embedded
library and persists to a local folder. Great for a single laptop.
"""
from __future__ import annotations

import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import config
from .document_processor import Chunk
from .embeddings import embed_texts, embedding_dim


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(path=config.QDRANT_PATH)
        self.collection = config.COLLECTION_NAME

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=embedding_dim(),
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def index_chunks(self, chunks: List[Chunk], batch_size: int = 64):
        self._ensure_collection()
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = embed_texts([c.text for c in batch])
            points = [
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors[j].tolist(),
                    payload={
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "doc_name": c.doc_name,
                        "page": c.page,
                        "text": c.text,
                        **c.metadata,
                    },
                )
                for j, c in enumerate(batch)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
        print(f"[vector_store] indexed {len(chunks)} chunks into Qdrant")

    def search(self, query: str, top_k: int = 10):
        self._ensure_collection()
        query_vec = embed_texts([query])[0].tolist()
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vec,
            limit=top_k,
        )
        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "doc_name": r.payload["doc_name"],
                "page": r.payload["page"],
                "text": r.payload["text"],
                "score": r.score,
            }
            for r in response.points
        ]

    def count(self) -> int:
        try:
            return self.client.count(collection_name=self.collection).count
        except Exception:
            return 0
