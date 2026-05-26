from __future__ import annotations

from langchain_core.documents import Document

from scripts.evaluate_retrieval import (
    RetrievalEvalCase,
    build_bucket_summary,
    build_retrieval_cases,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    summarize_metrics,
)


def test_retrieval_metrics_helpers():
    relevant = {"c1", "c2"}
    predicted = ["c3", "c2", "c1"]

    assert compute_recall_at_k(relevant, predicted, 2) == 0.5
    assert compute_mrr_at_k(relevant, predicted, 3) == 0.5
    assert round(compute_ndcg_at_k(relevant, predicted, 3), 4) == 0.6934


def test_build_retrieval_cases_uses_small_chunks_and_parent_relevance():
    docs = [
        Document(
            page_content="summary",
            metadata={"chunk_id": "summary-1", "chunk_level": "summary", "parent_id": "p1", "topic": "招生政策"},
        ),
        Document(
            page_content="small-1",
            metadata={
                "chunk_id": "c1",
                "chunk_level": "small",
                "parent_id": "p1",
                "topic": "招生政策",
                "source_title": "2025年招生章程",
                "section_hint": "收费标准",
                "chunk_text": "学费 5000 元",
                "query_expansions": ["收费标准 5000元", "学费标准"],
            },
        ),
        Document(
            page_content="small-2",
            metadata={
                "chunk_id": "c2",
                "chunk_level": "small",
                "parent_id": "p1",
                "topic": "招生政策",
                "source_title": "2025年招生章程",
                "section_hint": "住宿安排",
                "chunk_text": "住宿费 800 元",
                "query_expansions": ["住宿费标准"],
            },
        ),
    ]

    cases = build_retrieval_cases(docs=docs, max_cases=5)
    assert cases
    assert all(isinstance(case, RetrievalEvalCase) for case in cases)
    assert cases[0].category == "招生政策"
    assert "summary-1" in cases[0].relevant_chunk_ids
    assert "c2" in cases[0].relevant_chunk_ids
    assert {case.query_style for case in cases} >= {"semantic", "constraint", "lexical"}
    assert any("学费标准是多少" in case.query for case in cases)


def test_build_retrieval_cases_filters_noisy_raw_snippets():
    docs = [
        Document(
            page_content="summary",
            metadata={"chunk_id": "summary-2", "chunk_level": "summary", "parent_id": "p2", "topic": "学校概况"},
        ),
        Document(
            page_content="small",
            metadata={
                "chunk_id": "c3",
                "chunk_level": "small",
                "parent_id": "p2",
                "topic": "学校概况",
                "source_title": "学校概况",
                "section_hint": "# 原文（来源：https://example.com/abc）",
                "chunk_text": "# 原文（来源：https://example.com/abc） 网页标题：学校概况 抓取时间：2026-02-13 联系方式：0371-12345678",
                "query_expansions": [
                    "# 原文（来源：https://example.com/abc） 学校概况",
                    "学校概况 联系方式",
                ],
            },
        ),
    ]

    cases = build_retrieval_cases(docs=docs, max_cases=6)

    assert cases
    assert all("http" not in case.query for case in cases)
    assert all("原文（来源" not in case.query for case in cases)
    assert any("咨询电话" in case.query or "联系方式" in case.query for case in cases)


def test_summary_and_bucket_metrics():
    rows = [
        {"category": "招生政策", "recall@5": 1.0, "mrr@5": 0.5, "ndcg@5": 0.7, "latency_ms": 10},
        {"category": "招生政策", "recall@5": 0.5, "mrr@5": 0.25, "ndcg@5": 0.4, "latency_ms": 30},
        {"category": "费用资助", "recall@5": 1.0, "mrr@5": 1.0, "ndcg@5": 1.0, "latency_ms": 20},
    ]
    metrics = summarize_metrics(rows=rows, k=5)
    buckets = build_bucket_summary(rows=rows, k=5)

    assert metrics["recall@5"] == 0.8333
    assert metrics["mrr@5"] == 0.5833
    assert metrics["ndcg@5"] == 0.7
    assert metrics["p95_latency_ms"] == 30.0
    assert buckets["招生政策"]["total"] == 2
    assert buckets["招生政策"]["recall@5"] == 0.75
