from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from app.rag.index import RagIndexManager


@dataclass(slots=True)
class RetrievedItem:
    """统一召回结果结构，保留文档对象与融合分数。"""

    document: Document
    score: float


class HybridRetriever:
    """基于 BM25 + FAISS + EnsembleRetriever 的混合召回器。"""

    def __init__(self, index: RagIndexManager, sparse_weight: float = 0.35, dense_weight: float = 0.65):
        self.index = index
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

    def retrieve(self, queries: list[str], top_n: int) -> list[RetrievedItem]:
        # 关键变量：merged_scores 保存跨查询融合后的最终分数。
        merged_scores: dict[str, float] = {}
        merged_docs: dict[str, Document] = {}

        for query in queries:
            if not query.strip():
                continue
            rows = self._retrieve_single_query(query=query, top_n=top_n)
            for item in rows:
                chunk_id = str(item.document.metadata.get("chunk_id", ""))
                if not chunk_id:
                    continue
                merged_docs[chunk_id] = item.document
                merged_scores[chunk_id] = max(item.score, merged_scores.get(chunk_id, 0.0))

        ranked = [
            RetrievedItem(document=merged_docs[chunk_id], score=score)
            for chunk_id, score in merged_scores.items()
        ]
        ranked.sort(key=lambda row: row.score, reverse=True)
        return ranked[:top_n]

    def _retrieve_single_query(self, query: str, top_n: int) -> list[RetrievedItem]:
        bm25 = self.index.get_bm25_retriever(top_k=top_n)
        dense = self.index.get_dense_retriever(top_k=top_n)
        sparse_docs = bm25.invoke(query)
        dense_rows = self.index.dense_similarity_scores(query=query, top_k=top_n)
        dense_docs = [item[0] for item in dense_rows]

        try:
            from langchain.retrievers import EnsembleRetriever

            ensemble = EnsembleRetriever(
                retrievers=[bm25, dense],
                weights=[self.sparse_weight, self.dense_weight],
            )
            ensemble_docs = ensemble.invoke(query)
        except Exception:
            ensemble_docs = self._fallback_merge_order(sparse_docs=sparse_docs, dense_docs=dense_docs, top_n=top_n)

        # 关键变量：scores 把 rank 信息转换为可比较分数。
        scores = self._build_rank_score_map(
            sparse_docs=sparse_docs,
            dense_docs=dense_docs,
            ensemble_docs=ensemble_docs,
        )
        ranked: list[RetrievedItem] = []
        for doc in ensemble_docs:
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            if not chunk_id:
                continue
            ranked.append(RetrievedItem(document=doc, score=scores.get(chunk_id, 0.0)))
        return ranked[:top_n]

    def _build_rank_score_map(
        self,
        sparse_docs: list[Document],
        dense_docs: list[Document],
        ensemble_docs: list[Document],
    ) -> dict[str, float]:
        sparse_rank = self._rank_map(sparse_docs)
        dense_rank = self._rank_map(dense_docs)
        ensemble_rank = self._rank_map(ensemble_docs)
        scores: dict[str, float] = {}
        for chunk_id in set(sparse_rank) | set(dense_rank) | set(ensemble_rank):
            sparse_score = self.sparse_weight * (1.0 / (sparse_rank.get(chunk_id, 99) + 1))
            dense_score = self.dense_weight * (1.0 / (dense_rank.get(chunk_id, 99) + 1))
            ensemble_score = 0.1 * (1.0 / (ensemble_rank.get(chunk_id, 99) + 1))
            scores[chunk_id] = sparse_score + dense_score + ensemble_score
        return scores

    def _rank_map(self, docs: list[Document]) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for idx, doc in enumerate(docs):
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            if chunk_id and chunk_id not in mapping:
                mapping[chunk_id] = idx
        return mapping

    def _fallback_merge_order(self, sparse_docs: list[Document], dense_docs: list[Document], top_n: int) -> list[Document]:
        merged: list[Document] = []
        seen: set[str] = set()
        for row in [*dense_docs, *sparse_docs]:
            chunk_id = str(row.metadata.get("chunk_id", ""))
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(row)
            if len(merged) >= top_n:
                break
        return merged
