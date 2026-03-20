from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.rag.index import RagIndexManager
from app.rag.rerank import ListwiseReranker
from app.rag.retrievers import HybridRetriever
from app.rag.rewrite import QueryRewriter
from scripts.ablate_retrieval import _relevant_chunk_ids, build_variant_queries, evaluate_variant, summarize_variant
from scripts.evaluate_retrieval import RetrievalEvalCase, build_retrieval_cases


def _build_real_components(tmp_path: Path) -> tuple[RagIndexManager, QueryRewriter, HybridRetriever, ListwiseReranker]:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    sample = docs_dir / "01-招生政策.md"
    sample.write_text(
        "# 原文（来源：https://example.com/policy）\n"
        "网页标题：2025 招生政策\n"
        "抓取时间：2026-03-20\n"
        "发布时间：2025-05-15\n\n"
        "第一章 收费与资助\n"
        "第一条 学费 5000 元，住宿费 800 元。\n"
        "第二条 国家助学贷款每生每年最高 20000 元。\n"
        "第三条 新生报到流程包括资格核验、宿舍办理、缴费确认。\n"
        "第四条 咨询电话 0371-67698700。\n"
        "第五条 学费 5000 元，住宿费 800 元，奖助贷政策按年度通知执行。\n",
        encoding="utf-8",
    )
    settings = Settings(
        api_key="",
        api_url="https://example.com/v1/chat/completions",
        service_call_mode="local",
        use_mock_generation=True,
        docs_dir=docs_dir,
        rag_faiss_dir=tmp_path / "faiss",
        rag_chunk_size=80,
        rag_chunk_overlap=10,
        rag_retrieve_top_n=12,
        rag_final_top_k=4,
        rag_retry_top_n=16,
    )
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


def test_build_variant_queries_respects_rewrite_switch_with_real_rewriter(tmp_path: Path):
    _, rewriter, _, _ = _build_real_components(tmp_path)
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


def test_evaluate_variant_runs_rerank_variants_with_real_components(tmp_path: Path):
    index, rewriter, retriever, reranker = _build_real_components(tmp_path)
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


def test_relevant_chunk_ids_respects_small2big_switch_with_real_case(tmp_path: Path):
    index, _, _, _ = _build_real_components(tmp_path)
    cases = build_retrieval_cases(index.all_documents(), max_cases=8)
    case = next((item for item in cases if len(item.relevant_chunk_ids) > 1), None)

    assert case is not None
    assert len(_relevant_chunk_ids(case=case, variant="rrf_hybrid_small2big_on")) > 1
    assert len(_relevant_chunk_ids(case=case, variant="rrf_hybrid_small2big_off")) == 1
