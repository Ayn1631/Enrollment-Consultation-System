from __future__ import annotations

import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings


class QueryRewriter:
    """使用 LangChain 提供查询改写，失败时回退启发式策略。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm = self._build_llm()
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是招生检索改写器。先判断是否需要改写；无必要时保留原问。必须保留年份、省份、专业、分数、金额、否定词等硬约束。输出2-3条简短检索语句，每行一条，禁止解释。",
                ),
                ("user", "原始问题：{query}"),
            ]
        )

    def rewrite(self, query: str) -> list[str]:
        """改写用户问题，保证至少返回可检索语句。"""
        normalized = query.strip()
        if not normalized:
            return []
        if self._llm is None:
            return self._fallback_rewrite(normalized)

        try:
            chain = self._prompt | self._llm
            result = chain.invoke({"query": normalized})
            text = getattr(result, "content", str(result))
            rewritten = self._parse_lines(text)
            if rewritten:
                return rewritten
        except Exception:
            pass
        return self._fallback_rewrite(normalized)

    def _build_llm(self):
        """初始化改写模型，缺少依赖或密钥时返回 None。"""
        if self.settings.use_mock_generation or not self.settings.api_key:
            return None
        try:
            from langchain_openai import ChatOpenAI
        except Exception:
            return None
        base_url = self.settings.api_url.strip()
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=self.settings.api_key,
            base_url=base_url,
            temperature=0.1,
            timeout=self.settings.request_timeout_seconds,
        )

    def _parse_lines(self, text: str) -> list[str]:
        """解析模型输出行为去重查询列表。"""
        rows: list[str] = []
        seen: set[str] = set()
        for raw in text.splitlines():
            line = raw.strip(" -\t").strip()
            if not line:
                continue
            if line in seen:
                continue
            seen.add(line)
            rows.append(line)
            if len(rows) >= 3:
                break
        return rows

    def _fallback_rewrite(self, query: str) -> list[str]:
        """无可用 LLM 时，至少返回原问题和两个轻量变体。"""
        variants = [query]
        if self._needs_rewrite(query):
            suffixes = ["招生章程", "官方政策"]
            for suffix in suffixes:
                variants.append(f"{query} {suffix}".strip())
        else:
            variants.append(f"{query} 官方")
            constraint_tail = " ".join(self._extract_constraints(query))
            if constraint_tail:
                variants.append(f"{query} {constraint_tail}".strip())
            else:
                variants.append(f"{query} 招生")
        return list(dict.fromkeys(variants))[:3]

    def _needs_rewrite(self, query: str) -> bool:
        """简单判断是否需要扩展召回，而不是逮啥都改。"""
        if len(query) >= 18:
            return True
        if any(keyword in query for keyword in ("怎么", "如何", "能不能", "是否", "哪些", "多少")):
            return True
        return False

    def _extract_constraints(self, query: str) -> list[str]:
        """提取年份、数字和否定等硬约束，避免改写跑偏。"""
        constraints = re.findall(r"20\d{2}|\d+分|\d+元|不(?:要|能|可以|允许)|河南|河北|艺术类|理工类", query)
        return list(dict.fromkeys(constraints))
