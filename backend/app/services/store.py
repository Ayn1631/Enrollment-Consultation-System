from __future__ import annotations

import hashlib
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
    score: float = 0.0


class DocumentStore:
    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self._chunks: list[ChunkRecord] = []

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
                self._chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        title=title,
                        url=url,
                        text=normalized,
                    )
                )

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

    def search(self, query: str, top_k: int = 6) -> list[ChunkRecord]:
        tokens = [t for t in re.split(r"\s+", query) if t]
        scored: list[ChunkRecord] = []
        for chunk in self._chunks:
            score = 0.0
            for token in tokens:
                score += chunk.text.count(token) * 1.5
            if query and query in chunk.text:
                score += 8.0
            if score <= 0:
                continue
            scored.append(
                ChunkRecord(
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    url=chunk.url,
                    text=chunk.text,
                    score=score,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

