from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re

from langchain_core.documents import Document


class CitationGuard:
    """引用充分性校验器，决定是否允许给出确定性回答。"""

    def __init__(self, min_sources: int, min_top1_score: float):
        self.min_sources = min_sources
        self.min_top1_score = min_top1_score

    def validate(self, docs: list[Document]) -> tuple[bool, str | None]:
        """按来源数量与 top1 分数阈值判断是否通过引用校验。"""
        if not docs:
            return False, "no_source"
        top_score = float(docs[0].metadata.get("score", 0.0))
        source_count = len(
            {
                str(doc.metadata.get("source_url") or doc.metadata.get("source_title") or doc.metadata.get("doc_id"))
                for doc in docs
                if doc.metadata.get("source_url") or doc.metadata.get("source_title") or doc.metadata.get("doc_id")
            }
        )
        if source_count < self.min_sources:
            return False, "insufficient_sources"
        if top_score < self.min_top1_score:
            return False, "low_top_score"
        return True, None


@dataclass(slots=True)
class RetrievalQualityReport:
    passed: bool
    reason: str | None
    coverage: float
    unique_sources: int
    conflict_count: int
    stale_count: int


class RetrievalQualityGate:
    """检索质量闸门，拦截低覆盖或冲突证据进入生成阶段。"""

    def __init__(self, min_coverage: float = 0.25):
        self.min_coverage = min_coverage

    def evaluate(self, query: str, docs: list[Document]) -> RetrievalQualityReport:
        if not docs:
            return RetrievalQualityReport(
                passed=False,
                reason="empty_retrieval",
                coverage=0.0,
                unique_sources=0,
                conflict_count=0,
                stale_count=0,
            )

        coverage = self._coverage(query=query, docs=docs)
        unique_sources = len(
            {
                str(doc.metadata.get("source_url") or doc.metadata.get("source_title") or doc.metadata.get("doc_id"))
                for doc in docs
                if doc.metadata.get("source_url") or doc.metadata.get("source_title") or doc.metadata.get("doc_id")
            }
        )
        conflict_count = self._detect_conflicts(query=query, docs=docs)
        stale_count = self._count_stale_docs(docs)

        reason: str | None = None
        passed = True
        if conflict_count > 0:
            passed = False
            reason = "conflicting_evidence"
        elif coverage < self.min_coverage:
            passed = False
            reason = "low_coverage"
        elif self._is_time_sensitive(query) and stale_count >= len(docs):
            passed = False
            reason = "stale_evidence"

        return RetrievalQualityReport(
            passed=passed,
            reason=reason,
            coverage=coverage,
            unique_sources=unique_sources,
            conflict_count=conflict_count,
            stale_count=stale_count,
        )

    def _coverage(self, query: str, docs: list[Document]) -> float:
        tokens = self._tokenize(query)
        if not tokens:
            return 1.0
        corpus = " ".join(str(doc.metadata.get("chunk_text") or doc.page_content).lower() for doc in docs[:4])
        hits = sum(1 for token in tokens if token in corpus)
        return round(hits / max(len(tokens), 1), 4)

    def _detect_conflicts(self, query: str, docs: list[Document]) -> int:
        if not self._is_time_sensitive(query):
            return 0
        years: set[str] = set()
        for doc in docs[:4]:
            publish_date = str(doc.metadata.get("publish_date", "")).strip()
            if publish_date:
                years.add(publish_date[:4])
                continue
            topic = f"{doc.metadata.get('source_title', '')} {doc.metadata.get('topic', '')} {doc.page_content}"
            years.update(re.findall(r"\b(20\d{2})\b", topic))
        return 1 if len(years) > 1 else 0

    def _count_stale_docs(self, docs: list[Document]) -> int:
        today = date.today().isoformat()
        stale = 0
        for doc in docs:
            expire_date = str(doc.metadata.get("expire_date", "")).strip()
            if expire_date and expire_date < today:
                stale += 1
        return stale

    def _is_time_sensitive(self, query: str) -> bool:
        lowered = query.lower()
        keywords = ("最新", "当前", "现在", "今年", "最近", "近期")
        return any(word in lowered for word in keywords) or bool(re.search(r"\b20\d{2}\b", lowered))

    def _tokenize(self, text: str) -> list[str]:
        return [tok for tok in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+", text.lower()) if tok]
