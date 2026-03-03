from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import httpx

from app.models import FeatureFlag


# 关键变量：定义 Agent 功能执行优先级，保证引用校验在 RAG 后执行。
FEATURE_PRIORITY: dict[FeatureFlag, int] = {
    "rag": 1,
    "web_search": 2,
    "skill_exec": 3,
    "use_saved_skill": 4,
    "citation_guard": 5,
}


def _dedupe_features(features: list[FeatureFlag]) -> list[FeatureFlag]:
    """去重并保序，避免 Agent 重复执行同一功能。"""
    return list(dict.fromkeys(features))


class _PlanState(TypedDict):
    remaining: list[FeatureFlag]
    ordered: list[FeatureFlag]


class LangGraphFeaturePlanner:
    """使用 LangGraph 规划功能执行顺序，缺依赖时回退到本地排序。"""

    def plan(self, features: list[FeatureFlag]) -> list[FeatureFlag]:
        # 关键变量：normalized 保存去重后的输入，作为图执行初始状态。
        normalized = _dedupe_features(features)
        if not normalized:
            return []
        try:
            return self._plan_with_langgraph(normalized)
        except Exception:
            return self.fallback_plan(normalized)

    def _plan_with_langgraph(self, features: list[FeatureFlag]) -> list[FeatureFlag]:
        """通过 LangGraph 的 StateGraph 做可解释的执行计划。"""
        from langgraph.graph import END, StateGraph

        def arrange(state: _PlanState) -> _PlanState:
            remaining = list(state["remaining"])
            ordered = list(state["ordered"])
            if not remaining:
                return {"remaining": remaining, "ordered": ordered}

            # 关键变量：next_idx 按优先级和当前顺序联合排序，保证结果稳定。
            next_idx = min(
                range(len(remaining)),
                key=lambda idx: (FEATURE_PRIORITY.get(remaining[idx], 99), idx),
            )
            ordered.append(remaining.pop(next_idx))
            return {"remaining": remaining, "ordered": ordered}

        graph = StateGraph(_PlanState)
        graph.add_node("arrange", arrange)
        graph.set_entry_point("arrange")
        graph.add_conditional_edges("arrange", lambda s: END if not s["remaining"] else "arrange")
        compiled = graph.compile()
        result = compiled.invoke({"remaining": features, "ordered": []})
        return list(result["ordered"])

    def fallback_plan(self, features: list[FeatureFlag]) -> list[FeatureFlag]:
        """LangGraph 不可用时，按同一优先级策略本地降级。"""
        ordered_pairs = [(idx, item) for idx, item in enumerate(features)]
        ordered_pairs.sort(key=lambda pair: (FEATURE_PRIORITY.get(pair[1], 99), pair[0]))
        return [item for _, item in ordered_pairs]


@dataclass(slots=True)
class Neo4jKnowledgeAdapter:
    """从 Neo4j 查询与问题相关的知识图谱事实。"""

    uri: str
    user: str
    password: str
    database: str

    def enabled(self) -> bool:
        """只有 URI 和凭据齐全才尝试查询 Neo4j。"""
        return bool(self.uri and self.user and self.password)

    def fetch_facts(self, query: str, limit: int = 2) -> list[str]:
        """按查询词拉取图谱事实，异常时返回空列表并由网关降级。"""
        if not self.enabled():
            return []
        try:
            from neo4j import GraphDatabase
        except Exception:
            return []

        cypher = """
        CALL db.index.fulltext.queryNodes('admission_index', $query) YIELD node, score
        RETURN coalesce(node.name, node.title, '未知节点') AS name,
               coalesce(node.summary, node.text, '') AS summary,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        facts: list[str] = []
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with driver.session(database=self.database) as session:
                result = session.run(cypher, query=query, limit=limit)
                for row in result:
                    # 关键变量：fact_text 是写入上下文的统一字符串格式。
                    fact_text = f"{row.get('name', '')}: {row.get('summary', '')}".strip(": ").strip()
                    if fact_text:
                        facts.append(fact_text)
            driver.close()
        except Exception:
            return []
        return facts


@dataclass(slots=True)
class LangChain4jSkillBridge:
    """通过 HTTP 调用外部 LangChain4j 服务执行历史技能。"""

    base_url: str
    timeout_seconds: float

    def execute(self, query: str, session_id: str, saved_skill_id: str) -> str | None:
        """调用 LangChain4j 的技能端点，成功时返回说明文本。"""
        if not self.base_url or not saved_skill_id:
            return None

        endpoint = f"{self.base_url.rstrip('/')}/api/skills/execute"
        payload = {
            "query": query,
            "session_id": session_id,
            "saved_skill_id": saved_skill_id,
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                body = response.json()
        except Exception:
            return None

        note = body.get("note") or body.get("answer") or body.get("result")
        if not note:
            return None
        return str(note)
