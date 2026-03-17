from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re

from langchain_core.documents import Document

from app.rag.index import RagIndexManager


@dataclass(slots=True)
class RetrievedItem:
    """统一召回结果结构，保留文档对象与融合分数。"""

    document: Document
    score: float


class HybridRetriever:
    """基于 BM25 + Dense + Exact Match + RRF 的混合召回器。"""

    def __init__(
        self,
        index: RagIndexManager,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        exact_weight: float = 1.2,
        rrf_k: int = 60,
    ):
        self.index = index
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.exact_weight = exact_weight
        self.rrf_k = rrf_k

    def retrieve(self, queries: list[str], top_n: int) -> list[RetrievedItem]:
        """执行多查询召回并融合分数后返回 top_n。"""
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
        """执行单查询召回，并使用 RRF 融合三路排序。"""
        bm25 = self.index.get_bm25_retriever(top_k=top_n)
        sparse_docs = bm25.invoke(query)
        dense_rows = self.index.dense_similarity_scores(query=query, top_k=top_n)
        dense_docs = [item[0] for item in dense_rows]
        exact_docs = self._exact_match_search(query=query, top_n=top_n)

        scores = self._build_rank_score_map(
            sparse_docs=sparse_docs,
            dense_docs=dense_docs,
            exact_docs=exact_docs,
        )
        merged = self._merge_ranked_docs(
            candidates=[dense_docs, exact_docs, sparse_docs],
            scores=scores,
            query=query,
            top_n=top_n,
        )
        ranked_items: list[RetrievedItem] = []
        for doc in merged:
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            score = scores.get(chunk_id, 0.0) + self._temporal_bonus(query=query, doc=doc)
            ranked_items.append(RetrievedItem(document=doc, score=score))
        return ranked_items

    def _build_rank_score_map(self, sparse_docs: list[Document], dense_docs: list[Document], exact_docs: list[Document]) -> dict[str, float]:
        """使用 RRF 把多路排序位置映射为可比较分值。"""
        sparse_rank = self._rank_map(sparse_docs)
        dense_rank = self._rank_map(dense_docs)
        exact_rank = self._rank_map(exact_docs)
        scores: dict[str, float] = {}
        for chunk_id in set(sparse_rank) | set(dense_rank) | set(exact_rank):
            score = 0.0
            if chunk_id in sparse_rank:
                score += self.sparse_weight / (self.rrf_k + sparse_rank[chunk_id] + 1)
            if chunk_id in dense_rank:
                score += self.dense_weight / (self.rrf_k + dense_rank[chunk_id] + 1)
            if chunk_id in exact_rank:
                score += self.exact_weight / (self.rrf_k + exact_rank[chunk_id] + 1)
            scores[chunk_id] = score
        return scores

    def _rank_map(self, docs: list[Document]) -> dict[str, int]:
        """生成 chunk_id 到排名位置的映射。"""
        mapping: dict[str, int] = {}
        for idx, doc in enumerate(docs):
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            if chunk_id and chunk_id not in mapping:
                mapping[chunk_id] = idx
        return mapping

    def _merge_ranked_docs(self, candidates: list[list[Document]], scores: dict[str, float], query: str, top_n: int) -> list[Document]:
        """按融合分数合并候选文档，并做基础时效过滤与排序。"""
        seen: set[str] = set()
        ordered_candidates: list[Document] = []
        for rows in candidates:
            for row in rows:
                chunk_id = str(row.metadata.get("chunk_id", ""))
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                ordered_candidates.append(row)
        ordered_candidates = self._apply_temporal_policy(query=query, docs=ordered_candidates)
        ordered_candidates.sort(
            key=lambda doc: (
                scores.get(str(doc.metadata.get("chunk_id", "")), 0.0) + self._temporal_bonus(query=query, doc=doc),
                str(doc.metadata.get("publish_date", "")),
            ),
            reverse=True,
        )
        return ordered_candidates[:top_n]

    def _exact_match_search(self, query: str, top_n: int) -> list[Document]:
        """关键词精确匹配通道，补足对金额、年份、条款号等硬约束的召回。"""
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scored: list[tuple[float, Document]] = []
        normalized_query = query.strip().lower()
        for doc in self.index.all_documents():
            raw_text = " ".join(
                [
                    str(doc.metadata.get("chunk_text") or ""),
                    str(doc.metadata.get("source_title") or ""),
                    str(doc.metadata.get("topic") or ""),
                    str(doc.page_content),
                ]
            )
            text = raw_text.lower()
            phrase_hit = 1.5 if normalized_query and normalized_query in text else 0.0
            token_hits = sum(1 for token in tokens if token in text)
            if token_hits <= 0 and phrase_hit <= 0:
                continue
            coverage = token_hits / max(len(tokens), 1)
            numeric_hits = sum(1 for token in tokens if token.isdigit() and token in text)
            score = phrase_hit + coverage + numeric_hits * 0.25
            cloned = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            cloned.metadata["score"] = score
            scored.append((score, cloned))
        scored.sort(key=lambda item: (item[0], str(item[1].metadata.get("publish_date", ""))), reverse=True)
        return [item[1] for item in scored[:top_n]]

    def _tokenize(self, text: str) -> list[str]:
        """中英混合分词，供精确匹配通道复用。"""
        return [tok for tok in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+", text.lower()) if tok]

    def _apply_temporal_policy(self, query: str, docs: list[Document]) -> list[Document]:
        """对时效问题优先保留年份匹配或当前有效的证据。"""
        if not docs:
            return []
        query_years = set(re.findall(r"\b(20\d{2})\b", query))
        time_sensitive = self._is_time_sensitive(query)
        if not time_sensitive and not query_years:
            return docs

        if query_years:
            matched = [doc for doc in docs if self._doc_matches_year(doc, query_years)]
            if matched:
                return matched + [doc for doc in docs if doc not in matched]

        active_docs = [doc for doc in docs if self._is_active(doc)]
        if active_docs:
            return active_docs + [doc for doc in docs if doc not in active_docs]
        return docs

    def _temporal_bonus(self, query: str, doc: Document) -> float:
        """为时效匹配和有效期命中的文档增加排序偏置。"""
        query_years = set(re.findall(r"\b(20\d{2})\b", query))
        bonus = 0.0
        if query_years and self._doc_matches_year(doc, query_years):
            bonus += 1.2
        if self._is_time_sensitive(query) and self._is_active(doc):
            bonus += 0.6
        if self._is_expired(doc):
            bonus -= 0.8
        return bonus

    def _doc_matches_year(self, doc: Document, years: set[str]) -> bool:
        haystack = " ".join(
            [
                str(doc.metadata.get("publish_date", "")),
                str(doc.metadata.get("effective_date", "")),
                str(doc.metadata.get("expire_date", "")),
                str(doc.metadata.get("source_title", "")),
                str(doc.metadata.get("topic", "")),
                str(doc.metadata.get("chunk_text", "")),
            ]
        )
        return any(year in haystack for year in years)

    def _is_time_sensitive(self, query: str) -> bool:
        keywords = ("最新", "当前", "现在", "今年", "最近", "近期", "公告", "通知")
        return any(keyword in query for keyword in keywords) or bool(re.search(r"\b20\d{2}\b", query))

    def _is_active(self, doc: Document) -> bool:
        today = date.today().isoformat()
        effective_date = str(doc.metadata.get("effective_date", "")).strip()
        expire_date = str(doc.metadata.get("expire_date", "")).strip()
        if effective_date and effective_date > today:
            return False
        if expire_date and expire_date < today:
            return False
        return bool(effective_date or expire_date)

    def _is_expired(self, doc: Document) -> bool:
        expire_date = str(doc.metadata.get("expire_date", "")).strip()
        return bool(expire_date and expire_date < date.today().isoformat())
