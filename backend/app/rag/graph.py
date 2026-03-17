from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from langchain_core.documents import Document

from app.rag.citation_guard import CitationGuard
from app.rag.rerank import ListwiseReranker
from app.rag.retrievers import HybridRetriever
from app.rag.rewrite import QueryRewriter
from app.rag.types import RagGraphState


@dataclass(slots=True)
class RagGraphResult:
    trace_id: str
    status: str
    context_blocks: list[str]
    sources: list[dict[str, object]]
    degrade_reason: str | None
    latency_ms: dict[str, float]


class RagGraphOrchestrator:
    """LangGraph 版 RAG 工作流编排器。"""

    def __init__(
        self,
        rewriter: QueryRewriter,
        retriever: HybridRetriever,
        reranker: ListwiseReranker,
        citation_guard: CitationGuard,
        retrieve_top_n: int,
        final_top_k: int,
        node_timeout_ms: int,
    ):
        self.rewriter = rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.citation_guard = citation_guard
        self.retrieve_top_n = retrieve_top_n
        self.final_top_k = final_top_k
        self.node_timeout_ms = node_timeout_ms
        self._graph = self._compile_graph()

    def run(self, session_id: str, query: str, top_k: int) -> RagGraphResult:
        """执行图工作流并把状态投影为对外响应结构。"""
        initial_state: RagGraphState = {
            "trace_id": uuid.uuid4().hex,
            "session_id": dsession_id,
            "raw_query": query,
            "latency_breakdown_ms": {},
            "degrade_reason": None,
        }
        state = self._graph.invoke(initial_state)
        docs = list(state.get("reranked_docs") or state.get("retrieved_docs") or [])
        limited = docs[:top_k]
        sources = [self._to_source(doc) for doc in limited]
        status = "degraded" if state.get("degrade_reason") else "ok"
        return RagGraphResult(
            trace_id=str(state.get("trace_id", "")),
            status=status,
            context_blocks=list(state.get("final_context_blocks", [])),
            sources=sources,
            degrade_reason=state.get("degrade_reason"),
            latency_ms=dict(state.get("latency_breakdown_ms", {})),
        )

    def _compile_graph(self):
        """构建固定节点顺序的 LangGraph。"""
        from langgraph.graph import END, StateGraph

        graph = StateGraph(RagGraphState)
        graph.add_node("normalize_query", self._normalize_query)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("hybrid_retrieve", self._hybrid_retrieve)
        graph.add_node("dedupe_and_merge", self._dedupe_and_merge)
        graph.add_node("rerank_docs", self._rerank_docs)
        graph.add_node("citation_guard", self._citation_guard)
        graph.add_node("build_context", self._build_context)
        graph.add_node("finalize", self._finalize)
        graph.set_entry_point("normalize_query")
        graph.add_edge("normalize_query", "rewrite_query")
        graph.add_edge("rewrite_query", "hybrid_retrieve")
        graph.add_edge("hybrid_retrieve", "dedupe_and_merge")
        graph.add_edge("dedupe_and_merge", "rerank_docs")
        graph.add_edge("rerank_docs", "citation_guard")
        graph.add_edge("citation_guard", "build_context")
        graph.add_edge("build_context", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _normalize_query(self, state: RagGraphState) -> RagGraphState:
        """节点：清洗原始问题，产出 normalized_query。"""
        return self._timed(
            state,
            "normalize_query",
            lambda: {"normalized_query": str(state.get("raw_query", "")).strip()},
        )

    def _rewrite_query(self, state: RagGraphState) -> RagGraphState:
        """节点：生成 2-3 条检索改写。"""
        def _run() -> RagGraphState:
            normalized = str(state.get("normalized_query", ""))
            rewritten = self.rewriter.rewrite(normalized)
            if normalized and normalized not in rewritten:
                rewritten.insert(0, normalized)
            return {"rewritten_queries": rewritten[:3]}

        return self._timed(state, "rewrite_query", _run)

    def _hybrid_retrieve(self, state: RagGraphState) -> RagGraphState:
        """节点：执行混合召回，并把结果转为 Document 列表。"""
        def _run() -> RagGraphState:
            queries = list(state.get("rewritten_queries", []))
            if not queries:
                return {"retrieved_docs": []}
            rows = self.retriever.retrieve(queries=queries, top_n=self.retrieve_top_n)
            docs: list[Document] = []
            for item in rows:
                doc = Document(page_content=item.document.page_content, metadata=dict(item.document.metadata))
                doc.metadata["score"] = float(item.score)
                docs.append(doc)
            return {"retrieved_docs": docs}

        try:
            return self._timed(state, "hybrid_retrieve", _run)
        except Exception:
            return self._merge_state(state, {"retrieved_docs": [], "degrade_reason": "retrieval_error"})

    def _dedupe_and_merge(self, state: RagGraphState) -> RagGraphState:
        """节点：按 chunk_id 去重，避免重复证据污染重排。"""
        def _run() -> RagGraphState:
            docs = list(state.get("retrieved_docs", []))
            deduped: list[Document] = []
            seen: set[str] = set()
            for doc in docs:
                chunk_id = str(doc.metadata.get("chunk_id", ""))
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                deduped.append(doc)
            return {"retrieved_docs": deduped}

        return self._timed(state, "dedupe_and_merge", _run)

    def _rerank_docs(self, state: RagGraphState) -> RagGraphState:
        """节点：对候选文档重排，失败时沿用召回顺序并标记降级。"""
        def _run() -> RagGraphState:
            docs = list(state.get("retrieved_docs", []))
            query = str(state.get("normalized_query", ""))
            ranked, degraded = self.reranker.rerank(query=query, docs=docs, top_k=self.final_top_k)
            payload: RagGraphState = {"reranked_docs": ranked}
            if degraded and not state.get("degrade_reason"):
                payload["degrade_reason"] = "rerank_degraded"
            return payload

        return self._timed(state, "rerank_docs", _run)

    def _citation_guard(self, state: RagGraphState) -> RagGraphState:
        """节点：验证来源充分性，失败时触发保守回答路径。"""
        def _run() -> RagGraphState:
            docs = list(state.get("reranked_docs") or state.get("retrieved_docs") or [])
            passed, reason = self.citation_guard.validate(docs)
            payload: RagGraphState = {"guard_passed": passed}
            if not passed and not state.get("degrade_reason"):
                payload["degrade_reason"] = reason or "citation_guard_failed"
            return payload

        return self._timed(state, "citation_guard", _run)

    def _build_context(self, state: RagGraphState) -> RagGraphState:
        """节点：拼装最终给生成模型使用的上下文块。"""
        def _run() -> RagGraphState:
            docs = list(state.get("reranked_docs") or state.get("retrieved_docs") or [])
            top_docs = docs[: self.final_top_k]
            blocks = [doc.page_content for doc in top_docs]
            return {"final_context_blocks": blocks}

        return self._timed(state, "build_context", _run)

    def _finalize(self, state: RagGraphState) -> RagGraphState:
        """节点：预留收尾节点，保持图结构稳定。"""
        return self._timed(state, "finalize", lambda: {})

    def _timed(self, state: RagGraphState, node: str, fn):
        """包装节点执行，统一采集耗时并处理超时降级。"""
        start = time.perf_counter()
        patch = fn()
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        merged = self._merge_state(state, patch)
        # 关键变量：latency_breakdown_ms 记录节点级耗时，用于 debug 与评测。
        latency = dict(merged.get("latency_breakdown_ms", {}))
        latency[node] = elapsed_ms
        merged["latency_breakdown_ms"] = latency
        if elapsed_ms > self.node_timeout_ms and not merged.get("degrade_reason"):
            merged["degrade_reason"] = f"node_timeout:{node}"
        return merged

    def _merge_state(self, state: RagGraphState, patch: RagGraphState) -> RagGraphState:
        """合并节点 patch，保持状态字典不可变式更新。"""
        merged: RagGraphState = dict(state)
        merged.update(patch)
        return merged

    def _to_source(self, doc: Document) -> dict[str, object]:
        """把文档元数据映射为前端可展示的来源结构。"""
        return {
            "title": str(doc.metadata.get("source_title", "")),
            "url": str(doc.metadata.get("source_url", "")),
            "chunk_id": str(doc.metadata.get("chunk_id", "")),
            "score": float(doc.metadata.get("score", 0.0)),
        }
