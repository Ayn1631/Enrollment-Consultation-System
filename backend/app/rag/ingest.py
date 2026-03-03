from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document


SOURCE_PATTERN = re.compile(r"^# 原文（来源：(.*?)）$")


class RagIngestor:
    """负责把招生文档加载并切分为 LangChain Document。"""

    def __init__(self, docs_dir: Path, chunk_size: int, chunk_overlap: int):
        # 关键变量：docs_dir 指向招生语料目录。
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self) -> list[Document]:
        """加载 docs_dir 中的 markdown 文档并执行切块。"""
        rows: list[Document] = []
        if not self.docs_dir.exists():
            return rows

        for path in sorted(self.docs_dir.glob("*.md")):
            if path.name.lower() == "readme.md":
                continue
            content = path.read_text(encoding="utf-8")
            source_url = self._extract_source_url(content)
            chunks = self._split_text(content)
            for idx, chunk in enumerate(chunks, start=1):
                normalized = re.sub(r"\s+", " ", chunk).strip()
                if len(normalized) < 20:
                    continue
                chunk_id = f"{path.stem}-{idx}"
                rows.append(
                    Document(
                        page_content=normalized,
                        metadata={
                            "doc_id": path.stem,
                            "source_title": path.stem,
                            "source_url": source_url,
                            "chunk_id": chunk_id,
                        },
                    )
                )
        return rows

    def _extract_source_url(self, content: str) -> str:
        """从文档首行解析来源 URL。"""
        lines = content.splitlines()
        first_line = lines[0].strip() if lines else ""
        matched = SOURCE_PATTERN.match(first_line)
        if not matched:
            return ""
        return matched.group(1).strip()

    def _split_text(self, text: str) -> list[str]:
        """优先使用 LangChain RecursiveCharacterTextSplitter，失败时降级手动切块。"""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            )
            return splitter.split_text(text)
        except Exception:
            return self._manual_split_text(text)

    def _manual_split_text(self, text: str) -> list[str]:
        """切分器不可用时的兜底切块策略。"""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return chunks
