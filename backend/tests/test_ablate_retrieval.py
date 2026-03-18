from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

from scripts.ablate_retrieval import build_variant_queries, evaluate_variant, summarize_variant
from scripts.evaluate_retrieval import RetrievalEvalCase


class _FakeRewriter:
    def rewrite(self, query: str) -> list[str]:
        return [query, f"{query} 招生章程", f"{query} 官方政策"]


class _FakeRetriever:
    def __init__(self, chunk_ids: list[str]):
        self.chunk_ids = chunk_ids
        self.calls: list[dict] = []

    def retrieve(self, queries: list[str], top_n: int):
        self.calls.append({"queries": list(queries), "top_n": top_n})
        rows = []
        for rank, chunk_id in enumerate(self.chunk_ids[:top_n], start=1):
            rows.append(
                SimpleNamespace(
                    document=Document(
                        page_content=f"document-{chunk_id}",
                        metadata={"chunk_id": chunk_id},
                    ),
                    score=float(top_n - rank + 1),
                )
            )
        return rows


class _FakeReranker:
    def __init__(self):
        self.calls: list[dict] = []

    def rerank(self, query: str, docs: list[Document], top_k: int):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "chunk_ids": [str(doc.metadata.get("chunk_id", "")) for doc in docs],
            }
        )
        ordered = sorted(
            docs,
            key=lambda doc: (str(doc.metadata.get("chunk_id", "")) != "c2", -float(doc.metadata.get("score", 0.0))),
        )
        return ordered[:top_k], False


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


def test_evaluate_variant_rerank_off_skips_reranker():
    case = RetrievalEvalCase(
        name="c2-1",
        category="招生政策",
        query="住宿费",
        relevant_chunk_ids=["c2"],
    )
    retriever = _FakeRetriever(chunk_ids=["c1", "c2", "c3"])
    reranker = _FakeReranker()

    summary = evaluate_variant(
        retriever=retriever,
        rewriter=_FakeRewriter(),
        reranker=reranker,
        cases=[case],
        variant="rrf_hybrid_rerank_off",
        k=2,
    )

    assert retriever.calls == [{"queries": ["住宿费"], "top_n": 2}]
    assert reranker.calls == []
    assert summary["mrr@2"] == 0.5


def test_evaluate_variant_rerank_on_expands_candidates_and_reorders():
    case = RetrievalEvalCase(
        name="c2-1",
        category="招生政策",
        query="住宿费",
        relevant_chunk_ids=["c2"],
    )
    retriever = _FakeRetriever(chunk_ids=["c1", "c2", "c3"])
    reranker = _FakeReranker()

    summary = evaluate_variant(
        retriever=retriever,
        rewriter=_FakeRewriter(),
        reranker=reranker,
        cases=[case],
        variant="rrf_hybrid_rerank_on",
        k=2,
    )

    assert retriever.calls == [{"queries": ["住宿费"], "top_n": 12}]
    assert reranker.calls == [{"query": "住宿费", "top_k": 2, "chunk_ids": ["c1", "c2", "c3"]}]
    assert summary["mrr@2"] == 1.0


def test_evaluate_variant_small2big_switch_changes_relevance_scope():
    case = RetrievalEvalCase(
        name="c2-1",
        category="招生政策",
        query="助学贷款",
        relevant_chunk_ids=["c2", "summary-1"],
    )
    retriever = _FakeRetriever(chunk_ids=["summary-1", "c2"])
    reranker = _FakeReranker()

    small2big_off = evaluate_variant(
        retriever=retriever,
        rewriter=_FakeRewriter(),
        reranker=reranker,
        cases=[case],
        variant="rrf_hybrid_small2big_off",
        k=2,
    )
    small2big_on = evaluate_variant(
        retriever=retriever,
        rewriter=_FakeRewriter(),
        reranker=reranker,
        cases=[case],
        variant="rrf_hybrid_small2big_on",
        k=2,
    )

    assert small2big_off["mrr@2"] == 0.5
    assert small2big_on["mrr@2"] == 1.0
