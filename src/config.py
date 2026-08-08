"""
Central configuration for LexisGraph.
Reads everything from environment variables (loaded from .env) so you
never have to hardcode secrets or paths in code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root regardless of where the script is run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _get_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


class Config:
    # LLM
    LLM_PROVIDER = _get("LLM_PROVIDER", "anthropic").lower()
    ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL = _get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    OPENAI_API_KEY = _get("OPENAI_API_KEY")
    OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-4o-mini")
    GROQ_API_KEY = _get("GROQ_API_KEY")
    GROQ_MODEL = _get("GROQ_MODEL", "llama-3.1-8b-instant")

    # Storage paths (all local, no server needed)
    QDRANT_PATH = str(PROJECT_ROOT / _get("QDRANT_PATH", "./data/qdrant_db"))
    BM25_INDEX_PATH = str(PROJECT_ROOT / _get("BM25_INDEX_PATH", "./data/bm25_index.pkl"))
    KG_PATH = str(PROJECT_ROOT / _get("KG_PATH", "./data/knowledge_graph.gpickle"))
    COLLECTION_NAME = _get("COLLECTION_NAME", "legal_docs")

    # Models
    EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    RERANKER_MODEL = _get("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

    # Chunking
    CHUNK_SIZE = _get_int("CHUNK_SIZE", 800)
    CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 120)

    # Retrieval
    TOP_K_VECTOR = _get_int("TOP_K_VECTOR", 10)
    TOP_K_BM25 = _get_int("TOP_K_BM25", 10)
    TOP_K_RERANK = _get_int("TOP_K_RERANK", 5)
    RRF_K = _get_int("RRF_K", 60)


config = Config()
