from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOURCE_PATTERN = re.compile(r"^# 原文（来源：(.*?)）$")


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    title: str
    url: str
    text: str
    tokens: list[str]
    term_freq: dict[str, int]
    score: float = 0.0
    bm25_score: float = 0.0
    vector_score: float = 0.0
    keyword_score: float = 0.0


class DocumentStore:
    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self._chunks: list[ChunkRecord] = []
        self._idf: dict[str, float] = {}
        self._avg_len: float = 1.0

    @property
    def chunks(self) -> list[ChunkRecord]:
        return self._chunks

    def load(self) -> None:
        self._chunks = []
        seen_hashes: set[str] = set()
        if not self.docs_dir.exists():
            return
        for path in sorted(self.docs_dir.glob("*.md")):
            if path.name == "README.md":
                continue
            content = path.read_text(encoding="utf-8")
            title = path.stem
            url = self._extract_source_url(content) or ""
            paragraphs = self._split_paragraphs(content)
            for paragraph in paragraphs:
                normalized = re.sub(r"\s+", " ", paragraph).strip()
                if len(normalized) < 40:
                    continue
                hash_key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                if hash_key in seen_hashes:
                    continue
                seen_hashes.add(hash_key)
                chunk_id = f"{path.stem}-{len(self._chunks)+1}"
                tokens = self._tokenize(normalized)
                term_freq: dict[str, int] = {}
                for token in tokens:
                    term_freq[token] = term_freq.get(token, 0) + 1
                self._chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        title=title,
                        url=url,
                        text=normalized,
                        tokens=tokens,
                        term_freq=term_freq,
                    )
                )
        self._build_idf()

    def _extract_source_url(self, content: str) -> str | None:
        first = content.splitlines()[0] if content.splitlines() else ""
        match = SOURCE_PATTERN.match(first.strip())
        if not match:
            return None
        return match.group(1)

    def _split_paragraphs(self, content: str) -> Iterable[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
        for part in paragraphs:
            if len(part) <= 500:
                yield part
                continue
            start = 0
            while start < len(part):
                end = min(start + 480, len(part))
                yield part[start:end]
                if end == len(part):
                    break
                start = max(0, end - 70)

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        # Chinese chars + english words + numbers
        return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+", text)

    def _build_idf(self) -> None:
        if not self._chunks:
            self._idf = {}
            self._avg_len = 1.0
            return
        doc_freq: dict[str, int] = {}
        total_len = 0
        for chunk in self._chunks:
            total_len += max(1, len(chunk.tokens))
            for token in set(chunk.tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        n = len(self._chunks)
        self._idf = {
            token: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for token, df in doc_freq.items()
        }
        self._avg_len = total_len / n

    def _bm25_score(self, query_tokens: list[str], chunk: ChunkRecord) -> float:
        k1 = 1.5
        b = 0.75
        score = 0.0
        doc_len = max(1, len(chunk.tokens))
        for token in query_tokens:
            tf = chunk.term_freq.get(token, 0)
            if tf <= 0:
                continue
            idf = self._idf.get(token, 0.0)
            numer = tf * (k1 + 1)
            denom = tf + k1 * (1 - b + b * doc_len / self._avg_len)
            score += idf * (numer / denom)
        return score

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        freq: dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        vec: dict[str, float] = {}
        norm = 0.0
        for token, count in freq.items():
            weight = count * self._idf.get(token, 0.0)
            if weight == 0:
                continue
            vec[token] = weight
            norm += weight * weight
        norm = math.sqrt(norm) if norm > 0 else 1.0
        for token in list(vec.keys()):
            vec[token] = vec[token] / norm
        return vec

    def _cosine_score(self, query_vec: dict[str, float], chunk: ChunkRecord) -> float:
        chunk_vec = self._tfidf_vector(chunk.tokens)
        if not query_vec or not chunk_vec:
            return 0.0
        score = 0.0
        for token, q_weight in query_vec.items():
            score += q_weight * chunk_vec.get(token, 0.0)
        return score

    def _keyword_score(self, query: str, chunk_text: str) -> float:
        if not query:
            return 0.0
        if query in chunk_text:
            return 2.0 + len(query) / 50
        score = 0.0
        for word in re.split(r"\s+", query):
            if not word:
                continue
            score += min(chunk_text.count(word), 3) * 0.2
        return score

    def _rrf(self, rankings: list[list[ChunkRecord]], k: int = 60) -> dict[str, float]:
        combined: dict[str, float] = {}
        for rank_list in rankings:
            for idx, chunk in enumerate(rank_list):
                combined[chunk.chunk_id] = combined.get(chunk.chunk_id, 0.0) + 1.0 / (k + idx + 1)
        return combined

    def search(self, query: str, top_k: int = 6) -> list[ChunkRecord]:
        query_tokens = self._tokenize(query)
        query_vec = self._tfidf_vector(query_tokens)

        bm25_ranked: list[ChunkRecord] = []
        vector_ranked: list[ChunkRecord] = []
        keyword_ranked: list[ChunkRecord] = []

        for chunk in self._chunks:
            bm25 = self._bm25_score(query_tokens, chunk)
            vec = self._cosine_score(query_vec, chunk)
            keyword = self._keyword_score(query, chunk.text)
            if bm25 <= 0 and vec <= 0 and keyword <= 0:
                continue
            enriched = ChunkRecord(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                url=chunk.url,
                text=chunk.text,
                tokens=chunk.tokens,
                term_freq=chunk.term_freq,
                score=0.0,
                bm25_score=bm25,
                vector_score=vec,
                keyword_score=keyword,
            )
            bm25_ranked.append(enriched)
            vector_ranked.append(enriched)
            keyword_ranked.append(enriched)

        bm25_ranked.sort(key=lambda item: item.bm25_score, reverse=True)
        vector_ranked.sort(key=lambda item: item.vector_score, reverse=True)
        keyword_ranked.sort(key=lambda item: item.keyword_score, reverse=True)

        combined = self._rrf([bm25_ranked, vector_ranked, keyword_ranked], k=60)
        by_id = {item.chunk_id: item for item in bm25_ranked}
        final = []
        for chunk_id, score in combined.items():
            item = by_id[chunk_id]
            final.append(
                ChunkRecord(
                    chunk_id=item.chunk_id,
                    title=item.title,
                    url=item.url,
                    text=item.text,
                    tokens=item.tokens,
                    term_freq=item.term_freq,
                    score=score,
                    bm25_score=item.bm25_score,
                    vector_score=item.vector_score,
                    keyword_score=item.keyword_score,
                )
            )
        final.sort(key=lambda x: x.score, reverse=True)
        return final[:top_k]
