from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from app.config import Settings


class ListwiseReranker:
    """基于 LangChain listwise rerank 的重排器，失败时回退启发式重排。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._compressor = self._build_compressor()

    def rerank(self, query: str, docs: list[Document], top_k: int) -> tuple[list[Document], bool]:
        """重排候选文档，返回结果及是否发生降级。"""
        if not docs:
            return [], False
        if self._compressor is None:
            return self._fallback_rerank(query=query, docs=docs, top_k=top_k), True
        try:
            ranked = self._compressor.compress_documents(docs, query=query)
            return list(ranked)[:top_k], False
        except Exception:
            return self._fallback_rerank(query=query, docs=docs, top_k=top_k), True

    def _build_compressor(self):
        """构建 LangChain listwise rerank 组件。"""
        if self.settings.use_mock_generation or not self.settings.api_key:
            return None
        try:
            from langchain.retrievers.document_compressors import LLMListwiseRerank
            from langchain_openai import ChatOpenAI
        except Exception:
            return None

        base_url = self.settings.api_url.strip()
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=self.settings.api_key,
            base_url=base_url,
            temperature=0,
            timeout=self.settings.request_timeout_seconds,
        )
        return LLMListwiseRerank.from_llm(llm=llm, top_n=self.settings.rag_final_top_k)

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
