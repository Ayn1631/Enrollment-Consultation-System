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
                    "你是招生检索改写器。需要综合会话记忆线索、原始问题，生成适合知识库检索的查询。"
                    "必须保留年份、省份、专业、分数、金额、否定词等硬约束。"
                    "若原问包含多个并列意图，要拆成多个独立检索语句。输出2-6条，每行一条，禁止解释。",
                ),
                ("user", "会话记忆：\n{memory_hints}\n\n原始问题：{query}"),
            ]
        )

    def rewrite(self, query: str, memory_hints: list[str] | None = None) -> list[str]:
        """改写用户问题，保证至少返回可检索语句。"""
        normalized = query.strip()
        if not normalized:
            return []
        memory_hints = list(memory_hints or [])
        heuristic_queries = self._fallback_rewrite(normalized, memory_hints=memory_hints)
        if self._llm is None:
            return heuristic_queries

        try:
            chain = self._prompt | self._llm
            result = chain.invoke(
                {
                    "query": normalized,
                    "memory_hints": "\n".join(memory_hints[:3]) if memory_hints else "无",
                }
            )
            text = getattr(result, "content", str(result))
            rewritten = self._parse_lines(text)
            if rewritten:
                return self._merge_queries(heuristic_queries, rewritten)
        except Exception:
            pass
        return heuristic_queries

    def _build_llm(self):
        """初始化改写模型，缺少依赖或密钥时返回 None。"""
        llm_api_key = self.settings.resolve_llm_api_key()
        if self.settings.use_mock_generation or not llm_api_key:
            return None
        try:
            from langchain_openai import ChatOpenAI
        except Exception:
            return None
        base_url = self.settings.resolve_llm_api_url()
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=llm_api_key,
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
            if len(rows) >= 6:
                break
        return rows

    def _fallback_rewrite(self, query: str, memory_hints: list[str] | None = None) -> list[str]:
        """无可用 LLM 时，至少返回原问题和两个轻量变体。"""
        variants: list[str] = []
        enriched_query = self._enrich_query_with_memory(query, memory_hints or [])
        variants.append(enriched_query)
        if enriched_query != query:
            variants.append(query)
        variants.extend(self._split_multi_query(enriched_query))
        target_query = enriched_query
        if self._needs_rewrite(target_query):
            suffixes = ["招生章程", "官方政策"]
            for suffix in suffixes:
                variants.append(f"{target_query} {suffix}".strip())
        else:
            variants.append(f"{target_query} 官方")
            constraint_tail = " ".join(self._extract_constraints(target_query))
            if constraint_tail:
                variants.append(f"{target_query} {constraint_tail}".strip())
            else:
                variants.append(f"{target_query} 招生")
        return self._merge_queries([], variants)

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

    def _enrich_query_with_memory(self, query: str, memory_hints: list[str]) -> str:
        """短追问优先拼接会话记忆，让检索不再只拿代词乱撞。"""
        normalized = query.strip()
        if not normalized or not memory_hints or not self._needs_memory_enrichment(normalized):
            return normalized
        hint_text = " ".join(self._normalize_memory_hint(item) for item in memory_hints[:2]).strip()
        if not hint_text:
            return normalized
        return f"{normalized} {hint_text}".strip()

    def _needs_memory_enrichment(self, query: str) -> bool:
        return len(query) <= 18 or any(token in query for token in ("这个", "那个", "它", "还", "那", "呢", "吗"))

    def _normalize_memory_hint(self, hint: str) -> str:
        normalized = re.sub(r"\[[^\]]+\]", " ", hint)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized[:48]

    def _split_multi_query(self, query: str) -> list[str]:
        """对复合问题做轻量拆分，便于后续并发检索。"""
        keywords = [
            "学费",
            "住宿费",
            "住宿",
            "奖学金",
            "资助",
            "贷款",
            "电话",
            "地址",
            "报到",
            "报名",
            "录取",
            "分数线",
            "转专业",
            "选科",
            "学制",
        ]
        hits = [keyword for keyword in keywords if keyword in query]
        unique_hits = list(dict.fromkeys(hits))
        if len(unique_hits) <= 1:
            return []
        if any(token in query for token in ("多少", "收费", "费用")):
            tail = "是多少"
        elif any(token in query for token in ("怎么", "如何", "办理", "申请")):
            tail = "怎么办"
        elif "什么" in query:
            tail = "是什么"
        else:
            tail = ""
        constraints = " ".join(self._extract_constraints(query)[:2])
        sub_queries: list[str] = []
        for keyword in unique_hits:
            base = f"{keyword}{tail}".strip() if tail else keyword
            sub_queries.append(f"{base} {constraints}".strip())
        return sub_queries[:4]

    def _merge_queries(self, primary: list[str], secondary: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for candidate in [*primary, *secondary]:
            line = " ".join(candidate.split()).strip()
            if not line or line in seen:
                continue
            seen.add(line)
            merged.append(line)
            if len(merged) >= 6:
                break
        return merged
