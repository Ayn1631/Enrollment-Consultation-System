from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import re

from app.config import Settings
from app.rag.ingest import RagIngestor


class OpenAICompatibleEmbeddings(Embeddings):
    """OpenAI 兼容 Embedding 客户端，支持同源 /embeddings 端点。"""

    def __init__(self, endpoint: str, api_key: str, model: str, timeout_seconds: float, force_local: bool = False):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.force_local = force_local

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        rows = self._embed([text])
        return rows[0] if rows else []

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.force_local or not self.api_key:
            return [self._local_fallback_embedding(text) for text in texts]
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(self.endpoint, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
            data = body.get("data", [])
            if len(data) != len(texts):
                raise RuntimeError("embedding response size mismatch")
            return [list(item.get("embedding", [])) for item in data]
        except Exception:
            return [self._local_fallback_embedding(text) for text in texts]

    def _local_fallback_embedding(self, text: str) -> list[float]:
        """离线降级向量，保证无网络场景仍可构建 FAISS。"""
        seed = sum(ord(char) for char in text)
        dim = 32
        vector = []
        for idx in range(dim):
            value = ((seed + idx * 13) % 97) / 97.0
            vector.append(value)
        return vector


class RagIndexManager:
    """负责 RAG 索引生命周期：构建、加载、重建与统计。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.faiss_dir: Path = settings.rag_faiss_dir
        self._documents: list[Document] = []
        self._vectorstore = None
        self._bm25 = None
        self._local_dense_vectors: list[list[float]] = []
        self._indexed_at: datetime | None = None
        self._embeddings = OpenAICompatibleEmbeddings(
            endpoint=settings.resolve_embedding_api_url(),
            api_key=settings.api_key,
            model=settings.embedding_model,
            timeout_seconds=settings.request_timeout_seconds,
            force_local=settings.use_mock_generation,
        )
        self._ingestor = RagIngestor(
            docs_dir=settings.docs_dir,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

    def startup(self) -> None:
        """服务启动时加载或构建索引。"""
        if self._can_load_vectorstore():
            loaded = self._try_load_vectorstore()
            if loaded:
                self._build_bm25()
                return
        self.reindex()

    def reindex(self) -> dict[str, Any]:
        """强制重建索引并落盘。"""
        self._documents = self._ingestor.load_documents()
        self._build_vectorstore()
        self._build_bm25()
        self._indexed_at = datetime.utcnow()
        return {
            "status": "ok",
            "chunks": len(self._documents),
            "updated_at": self._indexed_at.isoformat(),
        }

    def stats(self) -> dict[str, Any]:
        """返回索引规模与配置元信息。"""
        return {
            "chunks": len(self._documents),
            "indexed_at": self._indexed_at.isoformat() if self._indexed_at else "",
            "embedding_model": self.settings.embedding_model,
            "faiss_dir": str(self.faiss_dir),
        }

    def get_bm25_retriever(self, top_k: int):
        if self._bm25 is None:
            raise RuntimeError("bm25 retriever is not initialized")
        self._bm25.k = top_k
        return self._bm25

    def get_dense_retriever(self, top_k: int):
        if self._vectorstore is not None:
            return self._vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

        manager = self

        class _LocalDenseRetriever:
            def invoke(self, query: str):
                rows = manager._local_similarity_search_with_scores(query=query, top_k=top_k)
                return [item[0] for item in rows]

        return _LocalDenseRetriever()

    def dense_similarity_scores(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        if self._vectorstore is not None:
            rows = self._vectorstore.similarity_search_with_score(query, k=top_k)
            normalized: list[tuple[Document, float]] = []
            for doc, raw_score in rows:
                relevance = 1.0 / (1.0 + max(float(raw_score), 0.0))
                normalized.append((doc, relevance))
            return normalized
        return self._local_similarity_search_with_scores(query=query, top_k=top_k)

    def _can_load_vectorstore(self) -> bool:
        return (self.faiss_dir / "index.faiss").exists() and (self.faiss_dir / "index.pkl").exists()

    def _try_load_vectorstore(self) -> bool:
        try:
            from langchain_community.vectorstores import FAISS
        except Exception:
            return False

        try:
            self._vectorstore = FAISS.load_local(
                str(self.faiss_dir),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            self._documents = self._extract_documents_from_vectorstore()
            self._indexed_at = datetime.utcnow()
            return True
        except Exception:
            self._vectorstore = None
            self._documents = []
            return False

    def _build_vectorstore(self) -> None:
        try:
            from langchain_community.vectorstores import FAISS
            self._vectorstore = FAISS.from_documents(self._documents, self._embeddings)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)
            self._vectorstore.save_local(str(self.faiss_dir))
            self._local_dense_vectors = []
            return
        except Exception:
            # FAISS 不可用时走本地向量检索降级，保持系统可用。
            self._vectorstore = None
            self._local_dense_vectors = self._embeddings.embed_documents([doc.page_content for doc in self._documents])

    def _build_bm25(self) -> None:
        try:
            from langchain_community.retrievers import BM25Retriever

            if not self._documents:
                self._bm25 = BM25Retriever.from_documents(
                    [Document(page_content="空语料", metadata={"chunk_id": "empty", "source_title": "empty", "source_url": ""})]
                )
                return
            self._bm25 = BM25Retriever.from_documents(self._documents)
            return
        except Exception:
            manager = self

            class _LocalBM25Retriever:
                def __init__(self):
                    self.k = manager.settings.rag_retrieve_top_n

                def invoke(self, query: str):
                    return manager._local_bm25_search(query=query, top_k=self.k)

            self._bm25 = _LocalBM25Retriever()

    def _extract_documents_from_vectorstore(self) -> list[Document]:
        if self._vectorstore is None:
            return []
        docs: list[Document] = []
        for doc_id in self._vectorstore.index_to_docstore_id.values():
            row = self._vectorstore.docstore.search(doc_id)
            if isinstance(row, Document):
                docs.append(row)
        return docs

    def _local_similarity_search_with_scores(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        if not self._documents or not self._local_dense_vectors:
            return []
        query_vec = self._embeddings.embed_query(query)
        scored: list[tuple[Document, float]] = []
        for doc, doc_vec in zip(self._documents, self._local_dense_vectors):
            score = self._cosine(query_vec, doc_vec)
            scored.append((doc, score))
        scored.sort(key=lambda row: row[1], reverse=True)
        return scored[:top_k]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _local_bm25_search(self, query: str, top_k: int) -> list[Document]:
        query_tokens = self._tokenize(query)
        scored: list[tuple[Document, float]] = []
        for doc in self._documents:
            text_tokens = self._tokenize(doc.page_content)
            overlap = 0.0
            for token in query_tokens:
                overlap += min(text_tokens.count(token), 4)
            score = overlap / max(1.0, len(text_tokens))
            cloned = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            cloned.metadata["score"] = score
            scored.append((cloned, score))
        scored.sort(key=lambda row: row[1], reverse=True)
        return [item[0] for item in scored[:top_k]]

    def _tokenize(self, text: str) -> list[str]:
        return [tok for tok in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+", text.lower()) if tok]
