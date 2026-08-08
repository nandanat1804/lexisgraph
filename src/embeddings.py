"""
Stage 2: Embeddings
Thin wrapper around a small ONNX-runtime embedding model (via `fastembed`),
NOT torch/sentence-transformers. Same underlying model
(sentence-transformers/all-MiniLM-L6-v2, ~90MB, 384-dim) and same output,
but ONNX runtime has a much smaller memory footprint than torch - this
matters if you're deploying on a free-tier host with ~512MB RAM (e.g.
Render's free plan), where torch + sentence-transformers alone can blow
the budget before your app even serves a request.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from fastembed import TextEmbedding

from .config import config


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    print(f"[embeddings] loading {config.EMBEDDING_MODEL} (ONNX/CPU, lightweight)...")
    return TextEmbedding(model_name=config.EMBEDDING_MODEL)


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    model = get_embedder()
    # fastembed returns L2-normalized vectors already (so cosine similarity
    # == dot product, same assumption the rest of the pipeline relies on)
    vectors = list(model.embed(texts, batch_size=batch_size))
    return np.array(vectors)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]


def embedding_dim() -> int:
    # all-MiniLM-L6-v2 is 384-dim; ask fastembed's registry so this stays
    # correct if EMBEDDING_MODEL is changed in .env
    from fastembed import TextEmbedding as TE
    for m in TE.list_supported_models():
        if m["model"] == config.EMBEDDING_MODEL:
            return m["dim"]
    # fallback: embed a throwaway string and check its length
    return len(embed_texts(["dimension probe"])[0])
