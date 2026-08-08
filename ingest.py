#!/usr/bin/env python3
"""
Run this once (or whenever you add new documents) to build the vector
index, BM25 index, and knowledge graph.

Usage:
    python ingest.py --docs sample_docs
    python ingest.py --docs /path/to/your/legal/pdfs
"""
import argparse

from src.rag_pipeline import LexisGraphPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest legal documents into LexisGraph")
    parser.add_argument("--docs", type=str, default="sample_docs",
                         help="Directory containing .pdf/.txt/.md legal documents")
    args = parser.parse_args()

    pipeline = LexisGraphPipeline()
    n_chunks = pipeline.ingest(args.docs)

    print("\n--- Ingestion summary ---")
    for k, v in pipeline.status().items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
