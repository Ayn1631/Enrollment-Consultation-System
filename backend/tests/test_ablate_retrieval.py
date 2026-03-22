from __future__ import annotations

from app.config import Settings
from app.rag.index import RagIndexManager
from app.rag.rerank import ListwiseReranker
from app.rag.retrievers import HybridRetriever
from app.rag.rewrite import QueryRewriter
from scripts.ablate_retrieval import _relevant_chunk_ids, build_variant_queries, evaluate_variant, summarize_variant
from scripts.evaluate_retrieval import RetrievalEvalCase, build_retrieval_cases


def _build_real_components(settings: Settings) -> tuple[RagIndexManager, QueryRewriter, HybridRetriever, ListwiseReranker]:
    index = RagIndexManager(settings)
    index.reindex()
    rewriter = QueryRewriter(settings)
    retriever = HybridRetriever(index=index)
    reranker = ListwiseReranker(settings)
    return index, rewriter, retriever, reranker


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


def test_build_variant_queries_respects_rewrite_switch_with_real_rewriter(isolated_runtime_settings: Settings):
    _, rewriter, _, _ = _build_real_components(isolated_runtime_settings)
    case = RetrievalEvalCase(
        name="c1-1",
        category="招生政策",
        query="学费 5000 元",
        relevant_chunk_ids=["c1"],
    )

    rewrite_off = build_variant_queries(case=case, variant="rrf_hybrid_rewrite_off", rewriter=rewriter)
    rewrite_on = build_variant_queries(case=case, variant="rrf_hybrid_rewrite_on", rewriter=rewriter)

    assert rewrite_off == ["学费 5000 元"]
    assert rewrite_on[0] == "学费 5000 元"
    assert len(rewrite_on) >= 2


def test_evaluate_variant_runs_rerank_variants_with_real_components(isolated_runtime_settings: Settings):
    index, rewriter, retriever, reranker = _build_real_components(isolated_runtime_settings)
    cases = build_retrieval_cases(index.all_documents(), max_cases=4)
    assert cases
    case = cases[0]

    rerank_off = evaluate_variant(
        retriever=retriever,
        rewriter=rewriter,
        reranker=reranker,
        cases=[case],
        variant="rrf_hybrid_rerank_off",
        k=2,
    )
    rerank_on = evaluate_variant(
        retriever=retriever,
        rewriter=rewriter,
        reranker=reranker,
        cases=[case],
        variant="rrf_hybrid_rerank_on",
        k=2,
    )

    for summary in (rerank_off, rerank_on):
        assert summary["cases"] == 1
        assert 0.0 <= float(summary["recall@2"]) <= 1.0
        assert 0.0 <= float(summary["mrr@2"]) <= 1.0
        assert 0.0 <= float(summary["ndcg@2"]) <= 1.0
        assert float(summary["avg_latency_ms"]) >= 0.0


def test_relevant_chunk_ids_respects_small2big_switch_with_real_case(isolated_runtime_settings: Settings):
    index, _, _, _ = _build_real_components(isolated_runtime_settings)
    cases = build_retrieval_cases(index.all_documents(), max_cases=8)
    case = next((item for item in cases if len(item.relevant_chunk_ids) > 1), None)

    assert case is not None
    assert len(_relevant_chunk_ids(case=case, variant="rrf_hybrid_small2big_on")) > 1
    assert len(_relevant_chunk_ids(case=case, variant="rrf_hybrid_small2big_off")) == 1
