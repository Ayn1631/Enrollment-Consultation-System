from __future__ import annotations

from langchain_core.documents import Document

from app.config import Settings
from app.rag.citation_guard import CitationGuard
from app.rag.graph import RagGraphOrchestrator
from app.rag.rerank import ListwiseReranker
from app.rag.retrievers import HybridRetriever
from app.rag.rewrite import QueryRewriter


class _FakeIndex:
    def get_bm25_retriever(self, top_k: int):
        class _Bm25:
            def __init__(self):
                self.k = top_k

            def invoke(self, query: str):
                return [
                    Document(page_content=f"{query} bm25-1", metadata={"chunk_id": "c1", "source_title": "T1", "source_url": "U1"}),
                    Document(page_content=f"{query} bm25-2", metadata={"chunk_id": "c2", "source_title": "T2", "source_url": "U2"}),
                ]

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
                Document(
                    page_content=f"{query} dense-score",
                    metadata={"chunk_id": "c3", "source_title": "T3", "source_url": "U3"},
                ),
                0.8,
            )
        ]


def test_query_rewriter_fallback_returns_multi_queries():
    settings = Settings(api_key="", api_url="https://example.com/v1/chat/completions", service_call_mode="local")
    rewriter = QueryRewriter(settings)
    rewriter._llm = None
    rows = rewriter.rewrite("招生章程")
    assert len(rows) >= 2
    assert rows[0] == "招生章程"


def test_hybrid_retriever_returns_dedup_ranked_docs():
    retriever = HybridRetriever(index=_FakeIndex())
    rows = retriever.retrieve(queries=["招生章程"], top_n=3)
    assert len(rows) >= 2
    assert rows[0].document.metadata["chunk_id"] in {"c1", "c2", "c3"}


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
    guard = CitationGuard(min_sources=3, min_top1_score=0.9)
    graph = RagGraphOrchestrator(
        rewriter=rewriter,
        retriever=retriever,
        reranker=reranker,
        citation_guard=guard,
        retrieve_top_n=5,
        final_top_k=3,
        node_timeout_ms=1200,
    )
    result = graph.run(session_id="s1", query="招生政策", top_k=3)
    assert result.trace_id
    assert result.status == "degraded"
    assert len(result.context_blocks) >= 1
