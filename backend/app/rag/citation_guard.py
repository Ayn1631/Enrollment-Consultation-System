from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
import re

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class CitationGuard:
    """引用充分性校验器，决定是否允许给出确定性回答。"""

    def __init__(self, min_sources: int, min_top1_score: float):
        self.min_sources = min_sources
        self.min_top1_score = min_top1_score

    def validate(self, docs: list[Document]) -> tuple[bool, str | None]:
        """按来源数量与 top1 分数阈值判断是否通过引用校验。"""
        if not docs:
            print("[CitationGuard] validate failed reason=no_source docs=0")
            logger.warning("[CitationGuard] validate failed reason=no_source docs=0")
            return False, "no_source"
        top_score = float(docs[0].metadata.get("score", 0.0))
        source_keys = [
            str(doc.metadata.get("source_url") or doc.metadata.get("source_title") or doc.metadata.get("doc_id"))
            for doc in docs
            if doc.metadata.get("source_url") or doc.metadata.get("source_title") or doc.metadata.get("doc_id")
        ]
        source_count = len(set(source_keys))
        message = (
            f"[CitationGuard] validate docs={len(docs)} top_score={top_score:.4f} "
            f"source_count={source_count} min_sources={self.min_sources} "
            f"min_top1_score={self.min_top1_score} sources={source_keys[:5]}"
        )
        print(message)
        logger.info(message)
        if source_count < self.min_sources:
            logger.warning("[CitationGuard] validate failed reason=insufficient_sources")
            return False, "insufficient_sources"
        if top_score < self.min_top1_score:
            logger.warning("[CitationGuard] validate failed reason=low_top_score")
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


@dataclass(slots=True)
class ConflictResolutionResult:
    docs: list[Document]
    resolved: bool
    note: str | None


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

    def resolve_conflicts(self, query: str, docs: list[Document]) -> ConflictResolutionResult:
        """按来源级别与时间优先级裁决冲突证据；裁不动则保留原结果。"""
        if not docs or not self._is_time_sensitive(query):
            return ConflictResolutionResult(docs=docs, resolved=False, note=None)

        scored_docs = [
            (
                self._conflict_priority(doc),
                idx,
                doc,
            )
            for idx, doc in enumerate(docs)
        ]
        if len({priority for priority, _, _ in scored_docs[:4]}) <= 1:
            return ConflictResolutionResult(docs=docs, resolved=False, note=None)

        scored_docs.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        best_priority, _, best_doc = scored_docs[0]
        second_priority = scored_docs[1][0] if len(scored_docs) > 1 else None
        if second_priority is not None and best_priority == second_priority:
            return ConflictResolutionResult(docs=docs, resolved=False, note=None)

        winner_year = self._doc_year(best_doc)
        winner_source = str(best_doc.metadata.get("source_url") or best_doc.metadata.get("source_title") or "")
        resolved_docs = [doc for _, _, doc in scored_docs if winner_year and self._doc_year(doc) == winner_year]
        if not resolved_docs:
            resolved_docs = [best_doc]
        return ConflictResolutionResult(
            docs=resolved_docs,
            resolved=True,
            note=f"resolved_conflict:{winner_year}:{winner_source}",
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

    def _conflict_priority(self, doc: Document) -> tuple[int, str, str]:
        """冲突裁决优先级：官方来源级别 > 生效时间 > 发布日期。"""
        return (
            self._source_authority(doc),
            str(doc.metadata.get("effective_date") or doc.metadata.get("publish_date") or self._doc_year(doc)),
            str(doc.metadata.get("publish_date") or self._doc_year(doc)),
        )

    def _source_authority(self, doc: Document) -> int:
        source = str(doc.metadata.get("source_url") or "").lower()
        if ".gov.cn" in source:
            return 4
        if "zut.edu.cn" in source or "zsc.zut.edu.cn" in source:
            return 3
        if ".edu.cn" in source:
            return 2
        if source:
            return 1
        return 0

    def _doc_year(self, doc: Document) -> str:
        for field in ("effective_date", "publish_date", "expire_date"):
            value = str(doc.metadata.get(field, "")).strip()
            if len(value) >= 4 and value[:4].isdigit():
                return value[:4]
        haystack = f"{doc.metadata.get('source_title', '')} {doc.metadata.get('topic', '')} {doc.page_content}"
        matched = re.search(r"\b(20\d{2})\b", haystack)
        return matched.group(1) if matched else ""
