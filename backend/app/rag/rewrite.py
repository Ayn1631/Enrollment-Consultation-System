from __future__ import annotations

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
                    "你是招生检索改写器。输出2-3条简短检索语句，每行一条，禁止解释。",
                ),
                ("user", "原始问题：{query}"),
            ]
        )

    def rewrite(self, query: str) -> list[str]:
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
        variants.append(f"{query} 招生章程")
        variants.append(f"{query} 官方政策")
        return list(dict.fromkeys(variants))[:3]
