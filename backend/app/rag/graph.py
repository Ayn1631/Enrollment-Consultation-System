from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
import re

from langchain_core.documents import Document

from app.rag.citation_guard import CitationGuard, RetrievalQualityGate
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
        quality_gate: RetrievalQualityGate,
        citation_guard: CitationGuard,
        retrieve_top_n: int,
        final_top_k: int,
        retry_top_n: int,
        node_timeout_ms: int,
    ):
        self.rewriter = rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.quality_gate = quality_gate
        self.citation_guard = citation_guard
        self.retrieve_top_n = retrieve_top_n
        self.final_top_k = final_top_k
        self.retry_top_n = max(retry_top_n, retrieve_top_n, final_top_k * 3)
        self.node_timeout_ms = node_timeout_ms
        self._graph = self._compile_graph()

    def run(self, session_id: str, query: str, top_k: int) -> RagGraphResult:
        """执行图工作流并把状态投影为对外响应结构。"""
        initial_state: RagGraphState = {
            "trace_id": uuid.uuid4().hex,
            "session_id": session_id,
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
        try:
            from langgraph.graph import END, StateGraph

            graph = StateGraph(RagGraphState)
            graph.add_node("normalize_query", self._normalize_query)
            graph.add_node("route_query", self._route_query)
            graph.add_node("rewrite_query", self._rewrite_query)
            graph.add_node("hybrid_retrieve", self._hybrid_retrieve)
            graph.add_node("dedupe_and_merge", self._dedupe_and_merge)
            graph.add_node("rerank_docs", self._rerank_docs)
            graph.add_node("quality_gate", self._quality_gate)
            graph.add_node("retry_retrieve", self._retry_retrieve)
            graph.add_node("citation_guard", self._citation_guard)
            graph.add_node("build_context", self._build_context)
            graph.add_node("finalize", self._finalize)
            graph.set_entry_point("normalize_query")
            graph.add_edge("normalize_query", "route_query")
            graph.add_edge("route_query", "rewrite_query")
            graph.add_edge("rewrite_query", "hybrid_retrieve")
            graph.add_edge("hybrid_retrieve", "dedupe_and_merge")
            graph.add_edge("dedupe_and_merge", "rerank_docs")
            graph.add_edge("rerank_docs", "quality_gate")
            graph.add_edge("quality_gate", "retry_retrieve")
            graph.add_edge("retry_retrieve", "citation_guard")
            graph.add_edge("citation_guard", "build_context")
            graph.add_edge("build_context", "finalize")
            graph.add_edge("finalize", END)
            return graph.compile()
        except Exception:
            return _LocalCompiledGraph(
                steps=[
                    self._normalize_query,
                    self._route_query,
                    self._rewrite_query,
                    self._hybrid_retrieve,
                    self._dedupe_and_merge,
                    self._rerank_docs,
                    self._quality_gate,
                    self._retry_retrieve,
                    self._citation_guard,
                    self._build_context,
                    self._finalize,
                ]
            )

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
            if state.get("route_label") == "follow_up":
                rewritten = [normalized] if normalized else []
            else:
                rewritten = self.rewriter.rewrite(normalized)
            if normalized and normalized not in rewritten:
                rewritten.insert(0, normalized)
            return {"rewritten_queries": rewritten[:3]}

        return self._timed(state, "rewrite_query", _run)

    def _route_query(self, state: RagGraphState) -> RagGraphState:
        """节点：根据问题类型选择检索深度，避免一把梭把延迟抬飞。"""
        def _run() -> RagGraphState:
            normalized = str(state.get("normalized_query", ""))
            route_label, route_reason = self._classify_route(normalized)
            top_n = self._route_top_n(route_label)
            summary_focus_parent_ids = self.retriever.locate_summary_parents(normalized, top_n=3) if route_label == "policy" else []
            return {
                "route_label": route_label,
                "route_reason": route_reason,
                "route_retrieve_top_n": top_n,
                "summary_focus_parent_ids": summary_focus_parent_ids,
            }

        return self._timed(state, "route_query", _run)

    def _hybrid_retrieve(self, state: RagGraphState) -> RagGraphState:
        """节点：执行混合召回，并把结果转为 Document 列表。"""
        def _run() -> RagGraphState:
            queries = list(state.get("rewritten_queries", []))
            if not queries:
                return {"retrieved_docs": []}
            retrieve_top_n = int(state.get("route_retrieve_top_n") or self.retrieve_top_n)
            rows = self.retriever.retrieve(
                queries=queries,
                top_n=retrieve_top_n,
                focus_parent_ids=list(state.get("summary_focus_parent_ids", [])),
            )
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

    def _quality_gate(self, state: RagGraphState) -> RagGraphState:
        """节点：对召回覆盖度、时效冲突做质量闸门控制。"""
        def _run() -> RagGraphState:
            docs = list(state.get("reranked_docs") or state.get("retrieved_docs") or [])
            report = self.quality_gate.evaluate(query=str(state.get("normalized_query", "")), docs=docs)
            payload: RagGraphState = {
                "quality_passed": report.passed,
                "quality_report": {
                    "passed": report.passed,
                    "reason": report.reason,
                    "coverage": report.coverage,
                    "unique_sources": report.unique_sources,
                    "conflict_count": report.conflict_count,
                    "stale_count": report.stale_count,
                },
            }
            if not report.passed and not state.get("degrade_reason"):
                payload["degrade_reason"] = report.reason or "quality_gate_failed"
            return payload

        return self._timed(state, "quality_gate", _run)

    def _retry_retrieve(self, state: RagGraphState) -> RagGraphState:
        """节点：低覆盖或空召回时执行一次更保守的二次检索。"""
        def _run() -> RagGraphState:
            report = dict(state.get("quality_report", {}))
            current_reason = str(report.get("reason") or state.get("degrade_reason") or "")
            retry_count = int(state.get("retry_count", 0))
            if retry_count >= 1 or current_reason not in {"low_coverage", "empty_retrieval", "stale_evidence"}:
                return {}

            query = str(state.get("normalized_query", ""))
            retry_queries = self._build_retry_queries(query=query, route_label=str(state.get("route_label", "")))
            try:
                rows = self.retriever.retrieve(
                    queries=retry_queries,
                    top_n=max(self.retry_top_n, int(state.get("route_retrieve_top_n") or 0)),
                    focus_parent_ids=list(state.get("summary_focus_parent_ids", [])),
                )
            except Exception:
                return {"retry_count": retry_count + 1, "degrade_reason": "retry_retrieval_failed"}
            docs: list[Document] = []
            for item in rows:
                doc = Document(page_content=item.document.page_content, metadata=dict(item.document.metadata))
                doc.metadata["score"] = float(item.score)
                docs.append(doc)
            ranked, degraded = self.reranker.rerank(query=query, docs=docs, top_k=self.final_top_k)
            quality_report = self.quality_gate.evaluate(query=query, docs=ranked or docs)
            payload: RagGraphState = {
                "retry_count": retry_count + 1,
                "retrieved_docs": docs,
                "reranked_docs": ranked,
                "quality_passed": quality_report.passed,
                "quality_report": {
                    "passed": quality_report.passed,
                    "reason": quality_report.reason,
                    "coverage": quality_report.coverage,
                    "unique_sources": quality_report.unique_sources,
                    "conflict_count": quality_report.conflict_count,
                    "stale_count": quality_report.stale_count,
                },
            }
            if degraded:
                payload["degrade_reason"] = "rerank_degraded"
            elif quality_report.passed and str(state.get("degrade_reason")) in {"low_coverage", "empty_retrieval", "stale_evidence"}:
                payload["degrade_reason"] = None
            elif not quality_report.passed:
                payload["degrade_reason"] = quality_report.reason or current_reason
            return payload

        return self._timed(state, "retry_retrieve", _run)

    def _build_context(self, state: RagGraphState) -> RagGraphState:
        """节点：拼装最终给生成模型使用的上下文块。"""
        def _run() -> RagGraphState:
            docs = list(state.get("reranked_docs") or state.get("retrieved_docs") or [])
            return {"final_context_blocks": self._arrange_context_blocks(docs)}

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

    def _classify_route(self, query: str) -> tuple[str, str]:
        """粗粒度问题路由，先解决有没有必要深检索。"""
        normalized = query.strip()
        if not normalized:
            return "policy", "empty_query"
        if any(keyword in normalized for keyword in ("最新", "当前", "现在", "今年", "公告", "通知", "最近", "近期")):
            return "time_sensitive", "time_sensitive_keyword"
        if len(normalized) <= 10 and any(keyword in normalized for keyword in ("还有", "那", "这个", "它", "呢", "吗")):
            return "follow_up", "short_follow_up"
        if any(keyword in normalized for keyword in ("电话", "地址", "住宿", "学费", "资助", "奖学金", "贷款")):
            return "faq", "faq_keyword"
        return "policy", "default_policy"

    def _route_top_n(self, route_label: str) -> int:
        """按问题类型控制检索深度。"""
        if route_label == "time_sensitive":
            return min(self.retrieve_top_n + 12, self.retry_top_n)
        if route_label == "follow_up":
            return max(12, self.retrieve_top_n // 2)
        if route_label == "faq":
            return max(20, self.retrieve_top_n - 8)
        return self.retrieve_top_n

    def _build_retry_queries(self, query: str, route_label: str) -> list[str]:
        """为二次检索构造更保守的查询集合。"""
        tokens = [tok for tok in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+", query.lower()) if tok]
        compact = "".join(tokens[:10])
        variants = [query]
        if compact and compact != query.lower():
            variants.append(compact)
        if route_label == "time_sensitive":
            variants.append(f"{query} 官方公告")
        else:
            variants.append(f"{query} 招生政策")
        return list(dict.fromkeys(item.strip() for item in variants if item.strip()))[:3]

    def _arrange_context_blocks(self, docs: list[Document]) -> list[str]:
        """执行去重、分组和首尾强化，减少长上下文中段被吃掉。"""
        grouped: dict[str, list[tuple[float, str, str]]] = {}
        seen_parents: set[str] = set()
        seen_hashes: set[str] = set()
        for doc in docs:
            parent_id = str(doc.metadata.get("parent_id", "")) or str(doc.metadata.get("chunk_id", ""))
            content_hash = str(doc.metadata.get("chunk_text_hash", "")) or str(doc.metadata.get("chunk_id", ""))
            if parent_id in seen_parents or content_hash in seen_hashes:
                continue
            seen_parents.add(parent_id)
            seen_hashes.add(content_hash)
            group_key = str(doc.metadata.get("topic") or doc.metadata.get("source_title") or "default")
            source_title = str(doc.metadata.get("source_title", "未命名来源"))
            block = str(doc.metadata.get("parent_text") or doc.page_content)
            grouped.setdefault(group_key, []).append((float(doc.metadata.get("score", 0.0)), source_title, block))

        ordered: list[str] = []
        for group_key, rows in sorted(grouped.items(), key=lambda item: max(row[0] for row in item[1]), reverse=True):
            rows.sort(key=lambda item: item[0], reverse=True)
            best_score, source_title, block = rows[0]
            ordered.append(f"[证据组:{group_key}][来源:{source_title}][score={best_score:.3f}]\n{block}")
            if len(ordered) >= self.final_top_k:
                break

        if len(ordered) <= 2:
            return ordered
        return [ordered[0], ordered[-1], *ordered[1:-1]][: self.final_top_k]


class _LocalCompiledGraph:
    """LangGraph 缺失时的本地顺序执行器。"""

    def __init__(self, steps):
        self.steps = steps

    def invoke(self, state: RagGraphState) -> RagGraphState:
        current = dict(state)
        for step in self.steps:
            current = step(current)
        return current
