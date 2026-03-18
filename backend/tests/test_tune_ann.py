from __future__ import annotations

from langchain_core.documents import Document

from scripts.tune_ann import AnnTuningRow, build_tuning_cases, compute_recall_at_k, summarize_curve


def test_build_tuning_cases_prefers_small_chunks_and_deduplicates_queries():
    docs = [
        Document(
            page_content="摘要层",
            metadata={
                "chunk_id": "summary-1",
                "chunk_level": "summary",
                "query_expansions": ["招生政策 官方规定"],
            },
        ),
        Document(
            page_content="正文",
            metadata={
                "chunk_id": "small-1",
                "chunk_level": "small",
                "chunk_text": "学费 5000 元，住宿费 800 元",
                "query_expansions": ["收费标准 5000元", "收费标准 5000元", "学费标准"],
            },
        ),
    ]

    cases = build_tuning_cases(docs=docs, max_cases=4)
    assert len(cases) >= 2
    assert all(case.target_chunk_id == "small-1" for case in cases)
    assert len({case.query for case in cases}) == len(cases)


def test_compute_recall_and_curve_summary():
    recall = compute_recall_at_k({"c1", "c2"}, ["c2", "c3", "c1"])
    assert recall == 1.0

    rows = [
        AnnTuningRow(m=16, ef_construction=64, ef_search=32, avg_recall_at_k=0.82, avg_latency_ms=1.5, p95_latency_ms=2.1),
        AnnTuningRow(m=16, ef_construction=64, ef_search=64, avg_recall_at_k=0.82, avg_latency_ms=1.8, p95_latency_ms=2.4),
        AnnTuningRow(m=32, ef_construction=128, ef_search=64, avg_recall_at_k=0.9, avg_latency_ms=2.3, p95_latency_ms=3.0),
    ]

    curve = summarize_curve(rows)
    assert len(curve) == 2
    assert curve[0]["avg_recall_at_k"] == 0.82
    assert curve[1]["avg_recall_at_k"] == 0.9
