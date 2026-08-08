#!/usr/bin/env python3
"""
Example evaluation run against the sample service agreement.
Edit EXAMPLES below to point at your own documents/questions.

Usage:
    python run_evaluation.py
"""
import json

from src.evaluation import EvalExample, evaluate_pipeline
from src.rag_pipeline import LexisGraphPipeline

EXAMPLES = [
    EvalExample(
        question="How much notice is required to terminate for convenience?",
        expected_doc_name="sample_service_agreement.txt",
        reference_answer="Either party may terminate for convenience with ninety (90) days written notice.",
    ),
    EvalExample(
        question="What is the cap on total liability?",
        expected_doc_name="sample_service_agreement.txt",
        reference_answer="Liability is capped at total fees paid in the preceding twelve months.",
    ),
    EvalExample(
        question="What law governs this agreement?",
        expected_doc_name="sample_service_agreement.txt",
        reference_answer="The Agreement is governed by the laws of the State of Delaware.",
    ),
    EvalExample(
        question="How long does the confidentiality obligation survive after termination?",
        expected_doc_name="sample_service_agreement.txt",
        reference_answer="The confidentiality obligation survives for five years after termination.",
    ),
]


def main():
    pipeline = LexisGraphPipeline()
    print("[eval] loading indexes (run ingest.py first if this fails)...")
    report = evaluate_pipeline(pipeline, EXAMPLES, top_k=5)
    print("\n=== Evaluation report ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
