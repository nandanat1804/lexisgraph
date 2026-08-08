#!/usr/bin/env python3
"""
Interactive command-line query loop against the indexed documents.

Usage:
    python query_cli.py
    python query_cli.py --question "What is the termination clause?"
"""
import argparse
import json

from src.rag_pipeline import LexisGraphPipeline


def print_result(result: dict):
    print("\n" + "=" * 70)
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for i, s in enumerate(result["sources"], 1):
        score = s.get("rerank_score")
        score_str = f"{score:.3f}" if score is not None else "n/a"
        print(f"  [{i}] {s['doc_name']} (p.{s['page']}, score={score_str})")
        print(f"      {s['text']}...")
    print(f"\nLatency: {result['latency_seconds']}s | "
          f"KG-related extra chunks found: {result['kg_related_chunk_count']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Query LexisGraph")
    parser.add_argument("--question", type=str, default=None,
                         help="Single question (skips interactive mode)")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output")
    args = parser.parse_args()

    pipeline = LexisGraphPipeline()

    if args.question:
        result = pipeline.query(args.question)
        print(json.dumps(result, indent=2)) if args.json else print_result(result)
        return

    print("LexisGraph interactive query mode. Type 'quit' to exit.\n")
    while True:
        question = input("Ask a legal question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        result = pipeline.query(question)
        print_result(result)


if __name__ == "__main__":
    main()
