from __future__ import annotations

from scripts.ablate_retrieval import summarize_variant


def test_summarize_variant_averages_metrics():
    rows = [
        {"recall@5": 1.0, "mrr@5": 0.5, "ndcg@5": 0.7, "latency_ms": 10.0},
        {"recall@5": 0.5, "mrr@5": 0.25, "ndcg@5": 0.4, "latency_ms": 30.0},
    ]

    summary = summarize_variant("rrf_hybrid", rows=rows, k=5)

    assert summary["variant"] == "rrf_hybrid"
    assert summary["cases"] == 2
    assert summary["recall@5"] == 0.75
    assert summary["mrr@5"] == 0.375
    assert summary["ndcg@5"] == 0.55
    assert summary["avg_latency_ms"] == 20.0
