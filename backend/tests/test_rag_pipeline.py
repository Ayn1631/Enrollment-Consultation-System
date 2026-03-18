from __future__ import annotations

from langchain_core.documents import Document

from app.config import Settings
from app.rag.citation_guard import CitationGuard, RetrievalQualityGate
from app.rag.ingest import RagIngestor
from app.rag.graph import RagGraphOrchestrator
from app.rag.rerank import ListwiseReranker
from app.rag.retrievers import HybridRetriever, RetrievedItem
from app.rag.rewrite import QueryRewriter


class _FakeIndex:
    def __init__(self):
        self._docs = [
            Document(
                page_content="标题：中原工学院2025年普通本科招生章程\n正文：学费 5000 元，住宿费 800 元",
                metadata={
                    "chunk_id": "c1",
                    "source_title": "T1",
                    "source_url": "U1",
                    "chunk_text": "学费 5000 元，住宿费 800 元",
                    "publish_date": "2025-05-15",
                },
            ),
            Document(
                page_content="标题：中原工学院2025年普通本科招生章程\n正文：国家助学贷款每生每年最高 20000 元",
                metadata={
                    "chunk_id": "c2",
                    "source_title": "T2",
                    "source_url": "U2",
                    "chunk_text": "国家助学贷款每生每年最高 20000 元",
                    "publish_date": "2025-05-15",
                },
            ),
            Document(
                page_content="标题：历史政策\n正文：2024 年学费说明",
                metadata={
                    "chunk_id": "c3",
                    "source_title": "T3",
                    "source_url": "U3",
                    "chunk_text": "2024 年学费说明",
                    "publish_date": "2024-05-15",
                    "effective_date": "2024-01-01",
                    "expire_date": "2024-12-31",
                },
            ),
            Document(
                page_content="标题：2026 就业公告\n正文：2026 春招公告发布",
                metadata={
                    "chunk_id": "c4",
                    "source_title": "T4",
                    "source_url": "U4",
                    "chunk_text": "2026 春招公告发布",
                    "publish_date": "2026-02-13",
                    "effective_date": "2026-02-13",
                },
            ),
        ]

    def get_bm25_retriever(self, top_k: int):
        docs = list(self._docs)

        class _Bm25:
            def __init__(self):
                self.k = top_k

            def invoke(self, query: str):
                return docs[: self.k]

        return _Bm25()

    def get_dense_retriever(self, top_k: int):
        class _Dense:
            def invoke(self, query: str):
                return [
                    Document(page_content=f"{query} dense-1", metadata={"chunk_id": "c2", "source_title": "T2", "source_url": "U2"}),
                    Document(page_content=f"{query} dense-2", metadata={"chunk_id": "c3", "source_title": "T3", "source_url": "U3"}),
                ]

        return _Dense()

    def dense_similarity_scores(self, query: str, top_k: int):
        return [
            (
                self._docs[2],
                0.8,
            ),
            (self._docs[1], 0.65),
        ]

    def all_documents(self):
        return list(self._docs)


def test_query_rewriter_fallback_returns_multi_queries():
    settings = Settings(api_key="", api_url="https://example.com/v1/chat/completions", service_call_mode="local")
    rewriter = QueryRewriter(settings)
    rewriter._llm = None
    rows = rewriter.rewrite("招生章程")
    assert len(rows) >= 2
    assert rows[0] == "招生章程"


def test_hybrid_retriever_returns_dedup_ranked_docs():
    retriever = HybridRetriever(index=_FakeIndex())
    rows = retriever.retrieve(queries=["助学贷款 20000 元"], top_n=3)
    assert len(rows) >= 2
    assert rows[0].document.metadata["chunk_id"] == "c2"


def test_hybrid_retriever_prefers_active_notice_for_latest_query():
    retriever = HybridRetriever(index=_FakeIndex())
    rows = retriever.retrieve(queries=["最新 2026 公告"], top_n=3)
    assert rows
    assert rows[0].document.metadata["chunk_id"] == "c4"


def test_listwise_reranker_fallback():
    settings = Settings(api_key="", api_url="https://example.com/v1/chat/completions", service_call_mode="local")
    reranker = ListwiseReranker(settings)
    reranker._compressor = None
    docs = [
        Document(page_content="学费和资助政策", metadata={"chunk_id": "c1", "score": 0.2}),
        Document(page_content="招生计划和批次说明", metadata={"chunk_id": "c2", "score": 0.4}),
    ]
    ranked, degraded = reranker.rerank(query="学费政策", docs=docs, top_k=2)
    assert degraded is True
    assert len(ranked) == 2


def test_citation_guard_thresholds():
    guard = CitationGuard(min_sources=2, min_top1_score=0.18)
    docs = [Document(page_content="x", metadata={"source_url": "u1", "score": 0.1})]
    ok, reason = guard.validate(docs)
    assert ok is False
    assert reason in {"low_top_score", "insufficient_sources"}


def test_rag_graph_runs_with_degrade_path(monkeypatch):
    settings = Settings(api_key="", api_url="https://example.com/v1/chat/completions", service_call_mode="local")
    rewriter = QueryRewriter(settings)
    rewriter._llm = None
    retriever = HybridRetriever(index=_FakeIndex())
    reranker = ListwiseReranker(settings)
    reranker._compressor = None
    quality_gate = RetrievalQualityGate(min_coverage=0.95)
    guard = CitationGuard(min_sources=3, min_top1_score=0.9)
    graph = RagGraphOrchestrator(
        rewriter=rewriter,
        retriever=retriever,
        reranker=reranker,
        quality_gate=quality_gate,
        citation_guard=guard,
        retrieve_top_n=5,
        final_top_k=3,
        retry_top_n=6,
        node_timeout_ms=1200,
    )
    result = graph.run(session_id="s1", query="招生政策", top_k=3)
    assert result.trace_id
    assert result.status == "degraded"
    assert len(result.context_blocks) >= 1


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


def test_rag_graph_retry_retrieve_recovers_low_coverage():
    class _RetryRetriever:
        def __init__(self):
            self.calls = 0

        def retrieve(self, queries: list[str], top_n: int, focus_parent_ids: list[str] | None = None):
            self.calls += 1
            if self.calls == 1:
                return [
                    RetrievedItem(
                        document=Document(
                            page_content="无关内容",
                            metadata={"chunk_id": "bad", "source_title": "T0", "source_url": "U0", "parent_text": "无关内容"},
                        ),
                        score=0.05,
                    )
                ]
            return [
                RetrievedItem(
                    document=Document(
                        page_content="标题：招生政策\n正文：学费 5000 元",
                        metadata={
                            "chunk_id": "good",
                            "source_title": "T1",
                            "source_url": "U1",
                            "parent_text": "标题：招生政策\n正文：学费 5000 元",
                            "chunk_text": "学费 5000 元",
                        },
                    ),
                    score=0.95,
                )
            ]

    class _PassThroughReranker:
        def rerank(self, query: str, docs: list[Document], top_k: int):
            return docs[:top_k], False

    settings = Settings(api_key="", api_url="https://example.com/v1/chat/completions", service_call_mode="local")
    rewriter = QueryRewriter(settings)
    rewriter._llm = None
    retriever = _RetryRetriever()
    graph = RagGraphOrchestrator(
        rewriter=rewriter,
        retriever=retriever,
        reranker=_PassThroughReranker(),
        quality_gate=RetrievalQualityGate(min_coverage=0.8),
        citation_guard=CitationGuard(min_sources=1, min_top1_score=0.1),
        retrieve_top_n=4,
        final_top_k=2,
        retry_top_n=6,
        node_timeout_ms=1200,
    )
    result = graph.run(session_id="s2", query="学费 5000", top_k=2)
    assert retriever.calls == 2
    assert result.status == "ok"
    assert any("学费 5000 元" in block for block in result.context_blocks)


def test_rag_graph_resolves_conflicting_versions_without_degrade():
    class _ConflictRetriever:
        def locate_summary_parents(self, query: str, top_n: int = 3):
            return []

        def retrieve(self, queries: list[str], top_n: int, focus_parent_ids: list[str] | None = None):
            docs = [
                Document(
                    page_content="标题：2025 招生章程\n正文：2025 年学费为 5000 元",
                    metadata={
                        "chunk_id": "n1",
                        "source_title": "2025招生章程",
                        "source_url": "https://zsc.zut.edu.cn/info/2025",
                        "chunk_text": "2025 年学费为 5000 元",
                        "publish_date": "2025-05-15",
                        "effective_date": "2025-01-01",
                        "parent_text": "标题：2025 招生章程\n正文：2025 年学费为 5000 元",
                    },
                ),
                Document(
                    page_content="标题：2024 招生章程\n正文：2024 年学费为 4800 元",
                    metadata={
                        "chunk_id": "o1",
                        "source_title": "2024招生章程",
                        "source_url": "https://zsc.zut.edu.cn/info/2024",
                        "chunk_text": "2024 年学费为 4800 元",
                        "publish_date": "2024-05-15",
                        "effective_date": "2024-01-01",
                        "parent_text": "标题：2024 招生章程\n正文：2024 年学费为 4800 元",
                    },
                ),
            ]
            return [RetrievedItem(document=doc, score=0.9 - idx * 0.1) for idx, doc in enumerate(docs)]

    class _PassThroughReranker:
        def rerank(self, query: str, docs: list[Document], top_k: int):
            return docs[:top_k], False

    settings = Settings(api_key="", api_url="https://example.com/v1/chat/completions", service_call_mode="local")
    graph = RagGraphOrchestrator(
        rewriter=QueryRewriter(settings),
        retriever=_ConflictRetriever(),
        reranker=_PassThroughReranker(),
        quality_gate=RetrievalQualityGate(min_coverage=0.1),
        citation_guard=CitationGuard(min_sources=1, min_top1_score=0.1),
        retrieve_top_n=4,
        final_top_k=2,
        retry_top_n=6,
        node_timeout_ms=1200,
    )
    graph.rewriter._llm = None
    result = graph.run(session_id="s3", query="最新招生章程", top_k=2)
    assert result.status == "ok"
    assert result.degrade_reason is None
    assert any("2025 年学费为 5000 元" in block for block in result.context_blocks)


def test_ingestor_extracts_metadata_and_parent_context(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    sample = docs_dir / "01-招生政策.md"
    sample.write_text(
        "# 原文（来源：https://example.com/policy）\n"
        "网页标题：2025 招生政策\n"
        "抓取时间：2026-02-13\n"
        "发布时间：2025-05-15\n\n"
        "第一章 总则\n"
        "第一条 学费 5000 元，住宿费 800 元。" * 20,
        encoding="utf-8",
    )
    ingestor = RagIngestor(docs_dir=docs_dir, chunk_size=80, chunk_overlap=10)
    docs = ingestor.load_documents()
    assert docs
    first = docs[0]
    assert first.metadata["publish_date"] == "2025-05-15"
    assert first.metadata["effective_date"] == "2025-01-01"
    assert first.metadata["expire_date"] == "2025-12-31"
    assert first.metadata["grab_date"] == "2026-02-13"
    assert first.metadata["topic"] == "招生政策"
    assert first.metadata["parent_id"].startswith("01-招生政策-parent-")
    assert "标题：2025 招生政策" in first.page_content
    assert first.metadata["chunk_text_hash"]


def test_ingestor_builds_summary_layer_and_locator_hits_parent(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    sample = docs_dir / "01-招生政策.md"
    sample.write_text(
        "# 原文（来源：https://example.com/policy）\n"
        "网页标题：2025 招生政策\n"
        "抓取时间：2026-02-13\n"
        "发布时间：2025-05-15\n\n"
        "第一章 总则\n"
        "第十七条 学费标准如下，理工类 5000 元，住宿费 800 元。\n"
        "第十八条 国家助学贷款每生每年最高 20000 元。\n",
        encoding="utf-8",
    )
    ingestor = RagIngestor(docs_dir=docs_dir, chunk_size=80, chunk_overlap=10)
    docs = ingestor.load_documents()
    summary_docs = [doc for doc in docs if doc.metadata.get("chunk_level") == "summary"]
    assert summary_docs

    class _Index:
        def all_documents(self):
            return docs

    retriever = HybridRetriever(index=_Index())
    parents = retriever.locate_summary_parents("助学贷款 20000 元", top_n=2)
    assert parents
    assert parents[0].startswith("01-招生政策-parent-")
