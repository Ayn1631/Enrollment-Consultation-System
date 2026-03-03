from __future__ import annotations

from langchain_core.documents import Document


class CitationGuard:
    """引用充分性校验器，决定是否允许给出确定性回答。"""

    def __init__(self, min_sources: int, min_top1_score: float):
        self.min_sources = min_sources
        self.min_top1_score = min_top1_score

    def validate(self, docs: list[Document]) -> tuple[bool, str | None]:
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
