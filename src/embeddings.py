"""
Stage 2: Embeddings
Thin wrapper around a small sentence-transformers model that runs fast
on CPU (~80MB, no GPU needed). Loaded once and reused (singleton).
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import config


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    print(f"[embeddings] loading {config.EMBEDDING_MODEL} (CPU)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    return model


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,  # so cosine similarity == dot product
        convert_to_numpy=True,
    )
    return vectors


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]


def embedding_dim() -> int:
    return get_embedder().get_sentence_embedding_dimension()
