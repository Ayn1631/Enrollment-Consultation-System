from __future__ import annotations

from app.config import Settings
from app.contracts import RagEvidence, RagQueryResponse
from app.rag.citation_guard import CitationGuard
from app.rag.graph import RagGraphOrchestrator
from app.rag.index import RagIndexManager
from app.rag.rerank import ListwiseReranker
from app.rag.retrievers import HybridRetriever
from app.rag.rewrite import QueryRewriter
from app.services.ai_stack import Neo4jKnowledgeAdapter


class RagGraphService:
    """RAG 统一服务入口，封装索引、检索、重排、图编排与图谱增强。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.index = RagIndexManager(settings)
        self.rewriter = QueryRewriter(settings)
        self.reranker = ListwiseReranker(settings)
        self.guard = CitationGuard(
            min_sources=settings.rag_citation_min_sources,
            min_top1_score=settings.rag_citation_min_top1_score,
        )
        self.retriever = HybridRetriever(index=self.index)
        self.orchestrator = RagGraphOrchestrator(
            rewriter=self.rewriter,
            retriever=self.retriever,
            reranker=self.reranker,
            citation_guard=self.guard,
            retrieve_top_n=settings.rag_retrieve_top_n,
            final_top_k=settings.rag_final_top_k,
            node_timeout_ms=settings.rag_node_timeout_ms,
        )
        self.neo4j_adapter = Neo4jKnowledgeAdapter(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )

    def startup(self) -> None:
        """初始化索引资源，通常在应用启动阶段调用。"""
        self.index.startup()

    def run(self, session_id: str, query: str, top_k: int = 8, debug: bool = False) -> RagQueryResponse:
        """执行一轮 RAG 查询并返回上下文、来源与降级状态。"""
        # 关键变量：effective_top_k 保证请求参数不会突破系统上限。
        effective_top_k = max(1, min(top_k, self.settings.rag_final_top_k))
        result = self.orchestrator.run(session_id=session_id, query=query, top_k=effective_top_k)
        sources = [
            RagEvidence(
                chunk_id=str(item["chunk_id"]),
                title=str(item["title"]),
                url=str(item["url"]),
                text="",
                score=float(item["score"]),
            )
            for item in result.sources
        ]
        context_blocks = list(result.context_blocks)
        if self.neo4j_adapter.enabled():
            facts = self.neo4j_adapter.fetch_facts(query=query, limit=2)
            for fact in facts:
                context_blocks.append(f"[neo4j] {fact}")
                sources.append(
                    RagEvidence(
                        chunk_id=f"neo4j-{len(sources)+1}",
                        title="Neo4j知识图谱",
                        url=self.settings.neo4j_uri,
                        text=fact,
                        score=0.12,
                    )
                )
        return RagQueryResponse(
            trace_id=result.trace_id,
            status=result.status,
            context_blocks=context_blocks,
            sources=sources,
            degrade_reason=result.degrade_reason,
            latency_ms=result.latency_ms if debug else {},
        )

    def reindex(self) -> dict:
        """触发索引重建。"""
        return self.index.reindex()

    def stats(self) -> dict:
        """返回索引统计信息。"""
        return self.index.stats()
