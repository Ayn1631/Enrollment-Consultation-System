from __future__ import annotations

from app.models import FeatureMeta, SavedSkill, ToolMeta


def feature_catalog() -> list[FeatureMeta]:
    return [
        FeatureMeta(id="rag", label="本地RAG检索", default_enabled=True),
        FeatureMeta(id="web_search", label="联网搜索增强", default_enabled=False),
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
    official_domains = ["zsc.zut.edu.cn", "zut.edu.cn"]
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
            id="web_search",
            label="官方站点联网搜索",
            kind="remote",
            timeout_seconds=0.8,
            retry_attempts=1,
            max_query_length=120,
            requires_time_sensitive=True,
            allowed_domains=official_domains,
            audit_scope="web_search",
        ),
        ToolMeta(
            id="web_read",
            label="官方网页阅读",
            kind="remote",
            timeout_seconds=1.0,
            retry_attempts=1,
            max_query_length=240,
            requires_time_sensitive=True,
            allowed_domains=official_domains,
            audit_scope="web_read",
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
    ]


def saved_skills() -> list[SavedSkill]:
    return [
        SavedSkill(id="admission_faq_v1", label="招生FAQ助手", description="聚焦招生政策与时间节点问答"),
        SavedSkill(id="fee_and_aid_v1", label="费用资助解读", description="学费、住宿费、奖助贷一体化解读"),
        SavedSkill(id="new_student_guide_v1", label="新生报到流程", description="报到、住宿、医保、校园服务引导"),
    ]
