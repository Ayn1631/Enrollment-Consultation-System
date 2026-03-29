from __future__ import annotations

from app.models import FeatureMeta, SavedSkill, ToolMeta


def feature_catalog() -> list[FeatureMeta]:
    return [
        FeatureMeta(id="rag", label="本地RAG检索", default_enabled=True),
        FeatureMeta(id="skill_exec", label="通用技能执行", default_enabled=False),
        FeatureMeta(
            id="use_saved_skill",
            label="使用以往技能",
            default_enabled=False,
            dependencies=["skill_exec"],
        ),
        FeatureMeta(id="citation_guard", label="引用校验", default_enabled=True, dependencies=["rag"]),
    ]


def tool_catalog() -> list[ToolMeta]:
    return [
        ToolMeta(
            id="local_rag",
            label="本地知识库检索",
            kind="local",
            timeout_seconds=1.2,
            retry_attempts=1,
            max_query_length=200,
            audit_scope="rag",
        ),
        ToolMeta(
            id="skill_exec",
            label="通用技能执行",
            kind="local",
            timeout_seconds=1.0,
            retry_attempts=1,
            max_query_length=200,
            audit_scope="skill_exec",
        ),
        ToolMeta(
            id="saved_skill",
            label="历史技能调用",
            kind="local",
            timeout_seconds=1.0,
            retry_attempts=1,
            max_query_length=200,
            audit_scope="use_saved_skill",
        ),
        ToolMeta(
            id="memory_recall",
            label="会话记忆读取",
            kind="local",
            timeout_seconds=0.4,
            retry_attempts=0,
            max_query_length=0,
            audit_scope="agent_memory",
        ),
        ToolMeta(
            id="mcp_tools_catalog",
            label="MCP工具目录",
            kind="local",
            timeout_seconds=0.2,
            retry_attempts=0,
            max_query_length=0,
            audit_scope="mcp_catalog",
        ),
        ToolMeta(
            id="mcp_tool_router",
            label="统一工具路由",
            kind="local",
            timeout_seconds=1.2,
            retry_attempts=1,
            max_query_length=200,
            audit_scope="agent_router",
        ),
    ]


def saved_skills() -> list[SavedSkill]:
    return [
        SavedSkill(id="admission_faq_v1", label="招生FAQ助手", description="聚焦招生政策与时间节点问答"),
        SavedSkill(id="fee_and_aid_v1", label="费用资助解读", description="学费、住宿费、奖助贷一体化解读"),
        SavedSkill(id="new_student_guide_v1", label="新生报到流程", description="报到、住宿、医保、校园服务引导"),
    ]
