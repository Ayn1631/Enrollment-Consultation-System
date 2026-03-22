from __future__ import annotations

import re
from typing import Any

import httpx
from langchain_core.documents import Document

from app.config import Settings


class ListwiseReranker:
    """基于 /rerank 接口的重排器，失败时回退启发式重排。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._endpoint = settings.resolve_rerank_api_url()
        self._api_key = settings.resolve_rerank_api_key()
        self._enabled = self._can_use_remote()

    def rerank(self, query: str, docs: list[Document], top_k: int) -> tuple[list[Document], bool]:
        """重排候选文档，返回结果及是否发生降级。"""
        if not docs:
            return [], False
        if not self._enabled:
            return self._fallback_rerank(query=query, docs=docs, top_k=top_k), True
        try:
            ranked = self._rerank_remote(query=query, docs=docs, top_k=top_k)
            if not ranked:
                raise RuntimeError("rerank response is empty")
            return ranked, False
        except Exception:
            return self._fallback_rerank(query=query, docs=docs, top_k=top_k), True

    def _can_use_remote(self) -> bool:
        return not self.settings.use_mock_generation and bool(self._api_key and self.settings.rerank_model)

    def _rerank_remote(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        payload: dict[str, Any] = {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": [self._document_text(doc) for doc in docs],
            "top_n": max(1, min(top_k, len(docs))),
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            response = client.post(self._endpoint, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        results = body.get("results", [])
        ranked: list[Document] = []
        for row in results:
            doc = self._map_result_to_document(row=row, docs=docs)
            if doc is not None:
                ranked.append(doc)
        return ranked[:top_k]

    def _map_result_to_document(self, row: dict[str, Any], docs: list[Document]) -> Document | None:
        try:
            index = int(row["index"])
        except (KeyError, TypeError, ValueError):
            return None
        if index < 0 or index >= len(docs):
            return None
        score = row.get("relevance_score", row.get("score", 0.0))
        cloned = Document(page_content=docs[index].page_content, metadata=dict(docs[index].metadata))
        cloned.metadata["score"] = float(score)
        return cloned

    def _document_text(self, doc: Document) -> str:
        chunk_text = str(doc.metadata.get("chunk_text", "")).strip()
        if chunk_text:
            return chunk_text
        return doc.page_content

    def _fallback_rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """回退排序：按关键词覆盖度与原分数进行混合排序。"""
        tokens = [tok for tok in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+", query.lower()) if tok]
        scored: list[tuple[float, Document]] = []
        for idx, doc in enumerate(docs):
            text = doc.page_content.lower()
            overlap = 0.0
            for token in tokens:
                overlap += min(text.count(token), 4) * 0.1
            base_score = float(doc.metadata.get("score", 0.0))
            recency_bonus = 1.0 / (idx + 1)
            publish_date = str(doc.metadata.get("publish_date", "")).strip()
            freshness = 0.03 if publish_date else 0.0
            score = base_score * 0.6 + overlap + recency_bonus * 0.05 + freshness
            cloned = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            cloned.metadata["score"] = score
            scored.append((score, cloned))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k]]
