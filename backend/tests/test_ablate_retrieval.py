from __future__ import annotations

from scripts.ablate_retrieval import build_variant_queries, summarize_variant
from scripts.evaluate_retrieval import RetrievalEvalCase


class _FakeRewriter:
    def rewrite(self, query: str) -> list[str]:
        return [query, f"{query} 招生章程", f"{query} 官方政策"]


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


def test_build_variant_queries_respects_rewrite_switch():
    case = RetrievalEvalCase(
        name="c1-1",
        category="招生政策",
        query="学费 5000 元",
        relevant_chunk_ids=["c1"],
    )
    rewriter = _FakeRewriter()

    rewrite_off = build_variant_queries(case=case, variant="rrf_hybrid_rewrite_off", rewriter=rewriter)
    rewrite_on = build_variant_queries(case=case, variant="rrf_hybrid_rewrite_on", rewriter=rewriter)

    assert rewrite_off == ["学费 5000 元"]
    assert rewrite_on[0] == "学费 5000 元"
    assert len(rewrite_on) == 3
