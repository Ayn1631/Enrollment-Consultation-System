from __future__ import annotations

import re

from app.contracts import RetrievalChunk


class SimpleReranker:
    def rerank(self, query: str, chunks: list[RetrievalChunk], top_k: int = 6) -> list[RetrievalChunk]:
        tokens = [token for token in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+", query.lower()) if token]
        rescored: list[tuple[float, RetrievalChunk]] = []
        for chunk in chunks:
            overlap = 0.0
            text = chunk.text.lower()
            for token in tokens:
                overlap += min(text.count(token), 4) * 0.12
            blended = chunk.score * 0.65 + overlap + chunk.vector_score * 0.2 + chunk.keyword_score * 0.15
            cloned = chunk.model_copy()
            cloned.score = blended
            rescored.append((blended, cloned))
        rescored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in rescored[:top_k]]

