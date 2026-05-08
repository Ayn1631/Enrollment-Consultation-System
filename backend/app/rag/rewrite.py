from __future__ import annotations

import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings


REWRITE_SYSTEM_PROMPT = """你是中原工学院招生信息检索的查询改写助手，任务是把用户问题改写成适合检索召回、且尽可能保留原始信息的自然中文查询句。

输入包含两部分：
1. 会话记忆：可能提供上一轮提到的年份、省份、专业、批次、科类、费用项目等上下文。
2. 原始问题：用户当前这一次的提问。

改写目标：
- 输出更完整、更适合检索的自然语言查询句，用于查询中原工学院招生政策、招生计划、录取分数、专业信息、收费标准、资助政策等官方信息。
- 如果原问题是短追问、代词追问、主语缺失问题，优先结合会话记忆补全主语和上下文。
- 如果原问题包含多个并列意图，必须拆分为相互独立、可单独检索的语句。
- 如果原问题只有一个意图，则输出 2 到 6 条围绕同一意图的高质量检索语句，表达可以有轻微变化，但语义和约束必须一致。
- 输出必须像用户会真的说出口或输入搜索框的人话，不要写成关键词堆砌、标签串、数据库条件或语法片段。

硬性约束：
- 必须完整保留并显式体现原问题中的年份、省份、专业、选科要求、批次、类别、分数、位次、金额、人群限定、否定词、比较对象等硬约束。
- 不得遗漏、改写、弱化或臆造任何硬约束。
- 不得把“这个专业”“那个专业”“去年那个分数线”“它”“这个”等指代词直接原样保留为最终主语；能从会话记忆中确定时必须补全。
- 如果会话记忆不足以补全主语，不要瞎编；应保留用户明确表达的检索目标，并用更通用但不失真的方式表述。
- 多个意图拆分后，每条查询只保留一个核心检索目标，但要带上与该目标相关的全部硬约束。
- 查询语句必须面向检索，但表现形式要像自然提问或自然表述，不要写成回答，不要出现解释性文字、推理过程、客套话或提示语。

改写规则：
- 优先把“学校 + 核心主题 + 关键约束”自然地组织进一句完整中文里，而不是机械拼词。
- 对口语化表达进行检索化改写，例如“多少分”“学费贵不贵”“能不能转专业”可以改成更利于检索的自然问法，但原意不能变。
- 对比较型问题，要在查询中保留比较双方和比较维度。
- 对否定约束必须原样保留，例如“不含中外合作办学”“不要专科”“不是艺术类”等。
- 对时间指代要尽量落到明确时间：若原问题或记忆中能确定“2024”“2025”等年份，则直接保留该年份；不能确定时不要擅自补年份。
- 输出语句应自然、简洁、像人话、可直接作为搜索查询。
- 不要输出类似“中原工学院 2024 河南 计算机专业 分数线”这种关键词串。
- 更倾向输出类似“中原工学院2024年在河南计算机科学与技术专业的录取分数线是多少”这种自然查询句。

输出要求：
- 仅输出 2 到 6 条查询。
- 每行一条。
- 不要添加解释、编号、标题、引号或任何额外内容。
- 不要输出重复语句。
- 每条都必须是自然中文查询句，允许是疑问句，也允许是简洁的陈述式查询句，但必须像正常人会说的话。

示例 1：
会话记忆：
- 上一轮在问 2024 年河南理科计算机科学与技术专业录取分数
原始问题：
- 那今年这个专业呢
正确输出示例：
中原工学院2025年在河南计算机科学与技术专业的录取分数线是多少
我想查中原工学院2025年河南计算机科学与技术专业的最低录取分数

示例 2：
会话记忆：
- 无
原始问题：
- 2023年河北考生报软件工程，不含中外合作办学，学费和最低分分别是多少
正确输出示例：
中原工学院2023年在河北招生的软件工程专业，不含中外合作办学的学费是多少
中原工学院2023年在河北招生的软件工程专业，不含中外合作办学的最低录取分数是多少

示例 3：
会话记忆：
- 上一轮提到河南物理类本科批
- 关注电气工程及其自动化专业
原始问题：
- 这个能不能转专业，宿舍一年多少钱，别给我中外合作的
正确输出示例：
中原工学院河南物理类电气工程及其自动化专业，不含中外合作办学的话，可以转专业吗
中原工学院河南物理类电气工程及其自动化专业，不含中外合作办学的话，宿舍一年多少钱
"""


class QueryRewriter:
    """使用 LangChain 提供查询改写，失败时回退启发式策略。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm = self._build_llm()
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REWRITE_SYSTEM_PROMPT),
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
