from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.config import Settings


def render_markdown_report(*, settings: Settings, report: dict[str, Any]) -> str:
    client = _build_client(settings)
    if client is None:
        return _fallback_markdown(report)
    try:
        response = client.chat.completions.create(
            model=settings.eval_judge_model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是招生 RAG 评测报告撰写助手。"
                        "请根据输入 JSON 生成一份中文 Markdown 报告。"
                        "必须包含：总览摘要、分类指标、路由表现、失败案例、改进建议。"
                        "只输出 Markdown，不要输出解释。"
                    ),
                },
                {"role": "user", "content": json.dumps(report, ensure_ascii=False)},
            ],
        )
        content = str(response.choices[0].message.content or "").strip()
        return content if content else _fallback_markdown(report)
    except Exception:
        return _fallback_markdown(report)


def _build_client(settings: Settings) -> OpenAI | None:
    api_key = settings.resolve_llm_api_key()
    base_url = settings.resolve_llm_api_url().rstrip("/")
    if not api_key or not base_url:
        return None
    base = base_url[:-len("/chat/completions")] if base_url.endswith("/chat/completions") else base_url
    return OpenAI(api_key=api_key, base_url=base, timeout=settings.llm_timeout_seconds)


def _fallback_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    route = report.get("route_summary", {})
    answer = report.get("answer_summary", {})
    failures = report.get("top_failures", [])
    buckets = report.get("bucket_summary", {})
    lines = [
        "# 招生资料 RAG 评测报告",
        "",
        "## 总览摘要",
        f"- 样本数：{summary.get('total', 0)}",
        f"- 通过率：{summary.get('pass_rate', 0)}",
        f"- 平均总分：{summary.get('avg_overall_score', 0)}",
        f"- Recall@5：{summary.get('recall@5', 0)}",
        f"- MRR@5：{summary.get('mrr@5', 0)}",
        f"- nDCG@5：{summary.get('ndcg@5', 0)}",
        "",
        "## 分类指标",
    ]
    for name, bucket in buckets.items():
        lines.append(f"- {name}：样本 {bucket.get('total', 0)}，通过率 {bucket.get('pass_rate', 0)}，平均分 {bucket.get('avg_overall_score', 0)}")
    lines.extend(
        [
            "",
            "## 路由表现",
            f"- 结构化路由命中率：{route.get('tool_route_accuracy', 0)}",
            f"- tool-first 正确率：{route.get('tool_first_pass_rate', 0)}",
            f"- rag-first 正确率：{route.get('rag_first_pass_rate', 0)}",
            "",
            "## 回答生成",
            f"- 回答失败率：{answer.get('answer_failure_rate', 0)}",
            f"- 回答模式分布：{json.dumps(answer.get('answer_modes', {}), ensure_ascii=False)}",
            "",
            "## 失败案例",
        ]
    )
    for row in failures[:10]:
        lines.append(f"- {row.get('case_id')}：{row.get('question')}，失败原因：{row.get('failure_reason')}")
    lines.extend(
        [
            "",
            "## 改进建议",
            "- 优先修复结构化路由误判与字段抽取缺失问题。",
            "- 对文本类失败样本补强 chunk 命中与引用一致性校验。",
            "- 对低分案例复查题目生成质量，避免无唯一证据的问题进入测试集。",
        ]
    )
    return "\n".join(lines).strip() + "\n"
