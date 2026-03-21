from __future__ import annotations

import httpx
from langchain_core.documents import Document
import pytest
import re

from app.config import Settings
from app.rag.citation_guard import CitationGuard, RetrievalQualityGate
from app.rag.service import RagGraphService
from app.rag.ingest import RagIngestor
from app.rag.index import RagIndexManager
from app.rag.rerank import ListwiseReranker
from app.rag.retrievers import HybridRetriever
from app.rag.rewrite import QueryRewriter


def _runtime_local_settings(isolated_runtime_settings: Settings, **updates) -> Settings:
    return isolated_runtime_settings.model_copy(
        update={
            "service_call_mode": "local",
            **updates,
        }
    )


def _build_real_index(settings: Settings) -> RagIndexManager:
    index = RagIndexManager(settings)
    index.reindex()
    return index


def test_query_rewriter_fallback_returns_multi_queries(isolated_runtime_settings: Settings):
    settings = _runtime_local_settings(isolated_runtime_settings)
    rewriter = QueryRewriter(settings)
    rewriter._llm = None
    rows = rewriter.rewrite("招生章程")
    assert len(rows) >= 2
    assert rows[0] == "招生章程"


def test_hybrid_retriever_returns_dedup_ranked_docs(isolated_runtime_settings: Settings):
    settings = _runtime_local_settings(isolated_runtime_settings)
    index = _build_real_index(settings)
    retriever = HybridRetriever(index=index)

    rows = retriever.retrieve(queries=["国家助学贷款每生每年最高不超过20000元"], top_n=6)

    assert rows
    chunk_ids = [str(row.document.metadata.get("chunk_id", "")) for row in rows]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert rows[0].document.metadata.get("source_title") == "中原工学院2025年普通本科招生章程"
    assert any("助学贷款" in str(row.document.metadata.get("chunk_text", "")) for row in rows[:3])
    assert any("20000" in str(row.document.metadata.get("chunk_text", "")) for row in rows[:3])


def test_hybrid_retriever_prefers_active_notice_for_latest_query(isolated_runtime_settings: Settings):
    settings = _runtime_local_settings(isolated_runtime_settings)
    index = _build_real_index(settings)
    retriever = HybridRetriever(index=index)

    rows = retriever.retrieve(queries=["最新 联系方式"], top_n=5)

    assert rows
    assert rows[0].document.metadata.get("source_title") == "联系方式"
    assert retriever._is_active(rows[0].document) is True  # noqa: SLF001
    assert str(rows[0].document.metadata.get("effective_date", "")).strip()
    assert any(row.document.metadata.get("source_title") == "联系方式" for row in rows[:3])


def test_listwise_reranker_fallback(isolated_runtime_settings: Settings):
    settings = _runtime_local_settings(
        isolated_runtime_settings,
        use_mock_generation=False,
        api_key="",
        llm_api_key="",
        rerank_api_key="",
    )
    index = _build_real_index(settings)
    reranker = ListwiseReranker(settings)
    docs = [doc for doc in index.all_documents() if doc.metadata.get("chunk_level") == "small"][:2]

    ranked, degraded = reranker.rerank(query="学费政策", docs=docs, top_k=2)

    assert degraded is True
    assert len(ranked) == 2
    assert all(doc.metadata.get("chunk_id") for doc in ranked)


def test_listwise_reranker_calls_remote_rerank_api(monkeypatch, runtime_settings: Settings):
    captured: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.97},
                    {"index": 0, "relevance_score": 0.41},
                ]
            }

    class _FakeClient:
        def __init__(self, timeout: float):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, headers: dict | None = None, json: dict | None = None):
            captured["url"] = url
            captured["headers"] = headers or {}
            captured["json"] = json or {}
            return _FakeResponse()

    monkeypatch.setattr("app.rag.rerank.httpx.Client", _FakeClient)
    if not runtime_settings.resolve_rerank_api_key():
        pytest.skip("当前环境未配置可用的 RERANK API KEY")
    settings = runtime_settings.model_copy(
        update={
            "use_mock_generation": False,
            "service_call_mode": "local",
        }
    )
    reranker = ListwiseReranker(settings)
    docs = [
        Document(page_content="学费和资助政策", metadata={"chunk_id": "c1", "chunk_text": "学费和资助政策"}),
        Document(page_content="招生计划和批次说明", metadata={"chunk_id": "c2", "chunk_text": "招生计划和批次说明"}),
    ]

    ranked, degraded = reranker.rerank(query="招生计划", docs=docs, top_k=2)

    assert degraded is False
    assert [doc.metadata["chunk_id"] for doc in ranked] == ["c2", "c1"]
    assert ranked[0].metadata["score"] == 0.97
    assert captured["url"] == settings.resolve_rerank_api_url()
    assert captured["headers"] == {
        "Authorization": f"Bearer {settings.resolve_rerank_api_key()}",
        "Content-Type": "application/json",
    }
    assert captured["json"] == {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "招生计划",
        "documents": ["学费和资助政策", "招生计划和批次说明"],
        "top_n": 2,
        "return_documents": False,
    }


def test_listwise_reranker_falls_back_when_remote_rerank_fails(monkeypatch, isolated_runtime_settings: Settings):
    class _FailingClient:
        def __init__(self, timeout: float):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, headers: dict | None = None, json: dict | None = None):
            raise httpx.ConnectError("rerank timeout")

    monkeypatch.setattr("app.rag.rerank.httpx.Client", _FailingClient)
    settings = _runtime_local_settings(isolated_runtime_settings, use_mock_generation=False)
    reranker = ListwiseReranker(settings)
    docs = [
        Document(page_content="学费和资助政策", metadata={"chunk_id": "c1", "score": 0.2}),
        Document(page_content="招生计划和批次说明", metadata={"chunk_id": "c2", "score": 0.4}),
    ]

    ranked, degraded = reranker.rerank(query="学费政策", docs=docs, top_k=2)

    assert degraded is True
    assert len(ranked) == 2
    assert ranked[0].metadata["chunk_id"] == "c1"


def test_citation_guard_thresholds():
    guard = CitationGuard(min_sources=2, min_top1_score=0.18)
    docs = [Document(page_content="x", metadata={"source_url": "u1", "score": 0.1})]
    ok, reason = guard.validate(docs)
    assert ok is False
    assert reason in {"low_top_score", "insufficient_sources"}


def test_rag_graph_runs_with_degrade_path(isolated_runtime_settings: Settings):
    settings = _runtime_local_settings(isolated_runtime_settings, rag_node_timeout_ms=5000)
    service = RagGraphService(settings)
    service.index.reindex()

    result = service.run(session_id="s1", query="最新招生章程", top_k=3, debug=True)

    assert result.trace_id
    assert result.status == "degraded"
    assert result.degrade_reason == "conflicting_evidence"
    assert len(result.context_blocks) >= 1
    assert result.sources


def test_retrieval_quality_gate_detects_conflicting_versions():
    gate = RetrievalQualityGate(min_coverage=0.1)
    docs = [
        Document(page_content="2025 招生章程", metadata={"source_url": "u1", "publish_date": "2025-05-15"}),
        Document(page_content="2024 招生章程", metadata={"source_url": "u2", "publish_date": "2024-05-15"}),
    ]
    report = gate.evaluate("2025 最新招生章程", docs)
    assert report.passed is False
    assert report.reason == "conflicting_evidence"


def test_retrieval_quality_gate_resolves_conflict_by_publish_date():
    gate = RetrievalQualityGate(min_coverage=0.1)
    docs = [
        Document(
            page_content="2025 招生章程",
            metadata={"source_url": "https://zsc.zut.edu.cn/info/1", "publish_date": "2025-05-15", "effective_date": "2025-01-01"},
        ),
        Document(
            page_content="2024 招生章程",
            metadata={"source_url": "https://zsc.zut.edu.cn/info/2", "publish_date": "2024-05-15", "effective_date": "2024-01-01"},
        ),
    ]
    result = gate.resolve_conflicts("最新招生章程", docs)
    assert result.resolved is True
    assert len(result.docs) == 1
    assert result.docs[0].metadata["publish_date"] == "2025-05-15"


def test_rag_graph_retry_retrieve_recovers_low_coverage(isolated_runtime_settings: Settings):
    settings = _runtime_local_settings(isolated_runtime_settings, rag_node_timeout_ms=5000)
    service = RagGraphService(settings)
    service.index.reindex()
    query = "国家助学贷款20000"

    state = {
        "trace_id": "retry-case",
        "session_id": "retry-case",
        "raw_query": query,
        "normalized_query": query,
        "route_label": "faq",
        "route_retrieve_top_n": settings.rag_retrieve_top_n,
        "summary_focus_parent_ids": service.retriever.locate_summary_parents(query, top_n=3),
        "quality_report": {
            "passed": False,
            "reason": "low_coverage",
            "coverage": 0.0,
            "unique_sources": 0,
            "conflict_count": 0,
            "stale_count": 0,
        },
        "degrade_reason": "low_coverage",
        "latency_breakdown_ms": {},
    }

    patched = service.orchestrator._retry_retrieve(state)  # noqa: SLF001

    assert patched["retry_count"] == 1
    assert patched["quality_passed"] is True
    assert patched["degrade_reason"] is None
    assert patched["reranked_docs"]
    assert any("助学贷款" in str(doc.metadata.get("chunk_text", "")) for doc in patched["reranked_docs"])


def test_rag_graph_precise_year_query_stays_usable(isolated_runtime_settings: Settings):
    settings = _runtime_local_settings(isolated_runtime_settings, rag_node_timeout_ms=5000)
    service = RagGraphService(settings)
    service.index.reindex()

    result = service.run(session_id="s3", query="国家助学贷款20000", top_k=2, debug=True)

    assert result.status == "ok"
    assert result.degrade_reason is None
    assert any(source.title == "中原工学院2025年普通本科招生章程" for source in result.sources)
    assert any("助学贷款" in block for block in result.context_blocks)


def test_ingestor_extracts_metadata_and_parent_context(runtime_docs_dir, runtime_settings: Settings):
    ingestor = RagIngestor(
        docs_dir=runtime_docs_dir,
        chunk_size=runtime_settings.rag_chunk_size,
        chunk_overlap=runtime_settings.rag_chunk_overlap,
    )
    docs = ingestor.load_documents()
    assert docs
    first = next((doc for doc in docs if doc.metadata.get("chunk_level") == "small"), None)
    assert first is not None
    assert first.metadata["source_title"]
    assert "parent-" in first.metadata["parent_id"]
    assert "标题：" in first.page_content
    assert first.metadata["chunk_text_hash"]
    small_docs = [doc for doc in docs if doc.metadata.get("chunk_level") == "small"]
    assert small_docs
    assert len(small_docs[0].metadata.get("query_expansions", [])) >= 2


def test_ingestor_builds_summary_layer_and_locator_hits_parent(runtime_docs_dir, runtime_settings: Settings):
    ingestor = RagIngestor(
        docs_dir=runtime_docs_dir,
        chunk_size=runtime_settings.rag_chunk_size,
        chunk_overlap=runtime_settings.rag_chunk_overlap,
    )
    docs = ingestor.load_documents()
    summary_docs = [doc for doc in docs if doc.metadata.get("chunk_level") == "summary"]
    assert summary_docs
    index = _build_real_index(runtime_settings)
    retriever = HybridRetriever(index=index)
    query = str(summary_docs[0].metadata.get("chunk_text", "")).split("；")[0].strip() or str(summary_docs[0].page_content)[:20]
    parents = retriever.locate_summary_parents(query, top_n=2)
    assert parents
    assert "parent-" in parents[0]


def test_bm25_uses_query_expansions_without_leaking_auxiliary_text(isolated_runtime_settings: Settings):
    settings = _runtime_local_settings(isolated_runtime_settings, use_mock_generation=True)
    index = RagIndexManager(settings)
    index.reindex()

    all_docs = index.all_documents()
    sample = next(
        (
            doc
            for doc in all_docs
            if doc.metadata.get("chunk_level") == "small"
            and doc.metadata.get("query_expansions")
        ),
        None,
    )
    assert sample is not None
    rows = index.get_bm25_retriever(top_k=3).invoke(str(sample.metadata.get("query_expansions", [""])[0]))
    assert rows
    matched = next((row for row in rows if row.metadata.get("chunk_level") == "small"), None)
    assert matched is not None
    assert matched.metadata.get("query_expansions")
    assert "辅助查询：" not in matched.page_content


def test_rag_result_for_real_docs_returns_matching_context_and_sources(isolated_runtime_settings: Settings):
    from app.rag.service import RagGraphService

    settings = _runtime_local_settings(isolated_runtime_settings)
    service = RagGraphService(settings)
    service.index.reindex()
    docs = service.index.all_documents()
    target = next(
        (
            doc
            for doc in docs
            if doc.metadata.get("chunk_level") == "small"
            and str(doc.metadata.get("chunk_text", "")).strip()
            and doc.metadata.get("query_expansions")
        ),
        None,
    )
    assert target is not None
    query = str(target.metadata.get("query_expansions", [target.metadata.get("chunk_text", "")])[0]).strip()
    expected_title = str(target.metadata.get("source_title", "")).strip()
    chunk_text = str(target.metadata.get("chunk_text", ""))
    matched = re.search(r"\d+(?:\.\d+)?(?:元|分|年|月|日|人|名|%)", chunk_text)
    probe_text = matched.group(0) if matched else re.sub(r"\s+", " ", chunk_text).strip()[:12]

    result = service.run(session_id="rag-real-docs-case", query=query, top_k=3, debug=True)

    assert result.trace_id
    assert result.context_blocks
    assert result.sources
    assert any(source.title == expected_title for source in result.sources)
    assert any(probe_text and probe_text in block for block in result.context_blocks)
    assert "rewrite_query" in result.latency_ms
