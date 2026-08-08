"""
Stage: Evaluation
Two things worth measuring in a RAG system:
1. Retrieval quality - did we fetch the right chunks? (Hit Rate, MRR)
2. Generation quality - does the answer match a reference? (ROUGE-L)

Usage: build a small eval set of (question, expected_doc_name,
reference_answer) tuples and run `evaluate_pipeline`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rouge_score import rouge_scorer

from .rag_pipeline import LexisGraphPipeline


@dataclass
class EvalExample:
    question: str
    expected_doc_name: str  # which source document should be retrieved
    reference_answer: str | None = None  # optional, for ROUGE-L


def hit_rate_at_k(retrieved_docs: List[str], expected_doc: str) -> int:
    return 1 if expected_doc in retrieved_docs else 0


def mrr(retrieved_docs: List[str], expected_doc: str) -> float:
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc == expected_doc:
            return 1.0 / rank
    return 0.0


def evaluate_pipeline(pipeline: LexisGraphPipeline, examples: List[EvalExample], top_k: int = 5) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    hit_rates, mrrs, rouge_ls, latencies = [], [], [], []

    for ex in examples:
        result = pipeline.query(ex.question, top_k=top_k)
        retrieved_doc_names = [s["doc_name"] for s in result["sources"]]

        hit_rates.append(hit_rate_at_k(retrieved_doc_names, ex.expected_doc_name))
        mrrs.append(mrr(retrieved_doc_names, ex.expected_doc_name))
        latencies.append(result["latency_seconds"])

        if ex.reference_answer:
            score = scorer.score(ex.reference_answer, result["answer"])
            rouge_ls.append(score["rougeL"].fmeasure)

    n = len(examples)
    report = {
        "num_examples": n,
        "hit_rate_at_k": sum(hit_rates) / n if n else 0,
        "mrr": sum(mrrs) / n if n else 0,
        "avg_latency_seconds": sum(latencies) / n if n else 0,
    }
    if rouge_ls:
        report["avg_rouge_l"] = sum(rouge_ls) / len(rouge_ls)
    return report
