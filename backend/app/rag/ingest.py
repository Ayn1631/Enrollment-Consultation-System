from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document


SOURCE_PATTERN = re.compile(r"^# 原文（来源：(.*?)）$")
TITLE_PATTERN = re.compile(r"^网页标题：(.+?)$")
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


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
        seen_hashes: set[tuple[str, str]] = set()
        if not self.docs_dir.exists():
            return rows

        for path in sorted(self.docs_dir.glob("*.md")):
            if path.name.lower() == "readme.md":
                continue
            content = path.read_text(encoding="utf-8")
            metadata = self._extract_doc_metadata(path=path, content=content)
            parent_chunks = self._split_text(
                content,
                chunk_size=max(self.chunk_size * 3, 1200),
                chunk_overlap=min(max(self.chunk_overlap * 2, 80), 160),
            )
            for parent_idx, parent_chunk in enumerate(parent_chunks, start=1):
                parent_text = self._build_contextual_chunk(parent_chunk, metadata)
                parent_id = f"{metadata['doc_id']}-parent-{parent_idx}"
                section_hint = self._extract_section_hint(parent_chunk)
                summary_doc = self._build_summary_document(
                    metadata=metadata,
                    parent_idx=parent_idx,
                    parent_id=parent_id,
                    parent_chunk=parent_chunk,
                    parent_text=parent_text,
                    section_hint=section_hint,
                )
                rows.append(summary_doc)
                child_chunks = self._split_text(
                    parent_chunk,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                for child_idx, chunk in enumerate(child_chunks, start=1):
                    normalized = re.sub(r"\s+", " ", chunk).strip()
                    if len(normalized) < 20:
                        continue
                    content_hash = self._normalized_hash(normalized)
                    dedupe_key = (str(metadata["source_url"]), content_hash)
                    if dedupe_key in seen_hashes:
                        continue
                    seen_hashes.add(dedupe_key)
                    chunk_id = f"{metadata['doc_id']}-{parent_idx}-{child_idx}"
                    contextualized = self._build_contextual_chunk(
                        chunk=normalized,
                        doc_meta=metadata,
                        section_hint=section_hint,
                    )
                    rows.append(
                        Document(
                            page_content=contextualized,
                            metadata={
                                **metadata,
                                "chunk_id": chunk_id,
                                "chunk_level": "small",
                                "parent_id": parent_id,
                                "parent_text": parent_text,
                                "section_hint": section_hint,
                                "chunk_text": normalized,
                                "chunk_text_hash": content_hash,
                            },
                        )
                    )
        return rows

    def _extract_doc_metadata(self, path: Path, content: str) -> dict[str, str]:
        """抽取统一 Schema 元数据，供 Small2Big 与引用追踪复用。"""
        lines = content.splitlines()
        first_line = lines[0].strip() if lines else ""
        matched = SOURCE_PATTERN.match(first_line)
        source_url = matched.group(1).strip() if matched else ""
        source_title = self._extract_title(path=path, lines=lines)
        publish_date = self._extract_date(lines=lines, prefix="发布时间：")
        grab_date = self._extract_date(lines=lines, prefix="抓取时间：")
        topic = self._extract_topic(path)
        effective_date, expire_date = self._derive_effective_window(
            source_title=source_title,
            topic=topic,
            publish_date=publish_date,
            content=content,
        )
        return {
            "doc_id": path.stem,
            "source_title": source_title,
            "source_url": source_url,
            "publish_date": publish_date,
            "effective_date": effective_date,
            "expire_date": expire_date,
            "grab_date": grab_date,
            "topic": topic,
        }

    def _extract_title(self, path: Path, lines: list[str]) -> str:
        """优先使用网页标题，缺失时回退文件名主题。"""
        for line in lines[:8]:
            matched = TITLE_PATTERN.match(line.strip())
            if matched:
                return matched.group(1).strip()
        return self._extract_topic(path)

    def _extract_topic(self, path: Path) -> str:
        """从文件名中提炼主题名称。"""
        return re.sub(r"^\d+-", "", path.stem).strip()

    def _extract_date(self, lines: list[str], prefix: str) -> str:
        """从指定前缀所在行解析日期。"""
        for line in lines[:12]:
            if not line.startswith(prefix):
                continue
            matched = DATE_PATTERN.search(line)
            if matched:
                return matched.group(1)
        return ""

    def _build_contextual_chunk(self, chunk: str, doc_meta: dict[str, str], section_hint: str | None = None) -> str:
        """为分块补充标题、主题和章节点摘要，增强短块可判读性。"""
        prefix_parts = [
            f"标题：{doc_meta['source_title']}",
            f"主题：{doc_meta['topic']}",
        ]
        if doc_meta.get("publish_date"):
            prefix_parts.append(f"发布时间：{doc_meta['publish_date']}")
        if doc_meta.get("effective_date"):
            prefix_parts.append(f"生效时间：{doc_meta['effective_date']}")
        if doc_meta.get("expire_date"):
            prefix_parts.append(f"失效时间：{doc_meta['expire_date']}")
        if section_hint:
            prefix_parts.append(f"章节点：{section_hint}")
        prefix = "\n".join(prefix_parts)
        return f"{prefix}\n正文：{chunk.strip()}".strip()

    def _build_summary_document(
        self,
        metadata: dict[str, str],
        parent_idx: int,
        parent_id: str,
        parent_chunk: str,
        parent_text: str,
        section_hint: str,
    ) -> Document:
        """为每个大块构建章节摘要层，供复杂问题先做导航定位。"""
        summary_text = self._summarize_parent_chunk(parent_chunk)
        summary_body = (
            f"标题：{metadata['source_title']}\n"
            f"主题：{metadata['topic']}\n"
            f"摘要层：{section_hint}\n"
            f"摘要内容：{summary_text}"
        )
        return Document(
            page_content=summary_body,
            metadata={
                **metadata,
                "chunk_id": f"{metadata['doc_id']}-summary-{parent_idx}",
                "chunk_level": "summary",
                "parent_id": parent_id,
                "parent_text": parent_text,
                "section_hint": section_hint,
                "chunk_text": summary_text,
                "chunk_text_hash": self._normalized_hash(summary_text),
            },
        )

    def _extract_section_hint(self, text: str) -> str:
        """从块内提取最有代表性的章节点提示。"""
        patterns = [
            r"(第[一二三四五六七八九十百零\d]+章[^\n。；]*)",
            r"(第[一二三四五六七八九十百零\d]+条[^\n。；]*)",
        ]
        for pattern in patterns:
            matched = re.search(pattern, text)
            if matched:
                return matched.group(1).strip()
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        return first_line[:40]

    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """优先使用 LangChain RecursiveCharacterTextSplitter，失败时降级手动切块。"""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            )
            return splitter.split_text(text)
        except Exception:
            return self._manual_split_text(text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _manual_split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """切分器不可用时的兜底切块策略。"""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - chunk_overlap)
        return chunks

    def _normalized_hash(self, text: str) -> str:
        """生成段落级去重哈希，避免重复内容污染召回。"""
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    def _summarize_parent_chunk(self, text: str) -> str:
        """抽取章节标题和关键句，构造轻量摘要层。"""
        normalized = [line.strip() for line in text.splitlines() if line.strip()]
        highlights: list[str] = []
        for line in normalized:
            if any(token in line for token in ("第", "学费", "资助", "录取", "招生", "电话", "地址", "奖学金", "贷款", "公告")):
                highlights.append(line)
            if len(highlights) >= 4:
                break
        if not highlights:
            highlights = normalized[:3]
        return "；".join(item[:80] for item in highlights if item)[:320]

    def _derive_effective_window(
        self,
        source_title: str,
        topic: str,
        publish_date: str,
        content: str,
    ) -> tuple[str, str]:
        """根据标题和年份线索推导知识有效期，支撑基础时效过滤。"""
        title_text = f"{source_title} {topic}"
        matched_years = YEAR_PATTERN.findall(title_text)
        if not matched_years:
            matched_years = YEAR_PATTERN.findall(content[:800])
        if publish_date and ("公告" in title_text or "通知" in title_text):
            return publish_date, ""
        if not matched_years:
            return "", ""
        year = matched_years[0]
        if any(keyword in title_text for keyword in ("章程", "政策", "招生", "录取", "资助")):
            return f"{year}-01-01", f"{year}-12-31"
        return "", ""
