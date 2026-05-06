from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(slots=True)
class StructuredRouteDecision:
    tool_name: str | None
    filters: dict[str, str]
    reason: str


def route_structured_query(query: str) -> StructuredRouteDecision:
    normalized = re.sub(r"\s+", " ", query).strip()
    if not normalized:
        return StructuredRouteDecision(tool_name=None, filters={}, reason="empty_query")

    if _is_scoreline_query(normalized):
        return StructuredRouteDecision(
            tool_name="scoreline_lookup",
            filters=_extract_scoreline_filters(normalized),
            reason="命中分数线/位次类关键词",
        )
    if _is_major_catalog_query(normalized):
        return StructuredRouteDecision(
            tool_name="major_catalog_lookup",
            filters=_extract_major_filters(normalized),
            reason="命中专业目录/学费/选科类关键词",
        )
    if _is_policy_table_query(normalized):
        return StructuredRouteDecision(
            tool_name="policy_table_lookup",
            filters=_extract_policy_filters(normalized),
            reason="命中政策附表类关键词",
        )
    return StructuredRouteDecision(tool_name=None, filters={}, reason="未命中结构化工具路由")


def _is_scoreline_query(query: str) -> bool:
    return any(token in query for token in ("分数线", "最低分", "位次", "投档", "录取分", "录取位次"))


def _is_major_catalog_query(query: str) -> bool:
    major_tokens = ("学费", "选考", "选科", "学制", "学位", "专业代码", "专业目录")
    if any(token in query for token in ("招生章程", "录取规则", "预留计划", "调档比例")):
        return False
    return any(token in query for token in major_tokens)


def _is_policy_table_query(query: str) -> bool:
    return any(token in query for token in ("专业情况汇总表", "政策附表", "章程附表", "汇总表"))


def _extract_major_filters(query: str) -> dict[str, str]:
    filters: dict[str, str] = {}
    year = _extract_year(query)
    if year:
        filters["academic_year"] = year
    code_match = re.search(r"\b([0-9A-Z]{4,8})\b", query)
    if code_match:
        filters["major_code"] = code_match.group(1)
    tuition = re.findall(r"(\d{4,5})", query)
    if "以上" in query and tuition:
        filters["tuition_min"] = tuition[0]
    elif "以下" in query and tuition:
        filters["tuition_max"] = tuition[0]
    for keyword in ("物理+化学", "不限", "物理", "历史"):
        if keyword in query:
            filters["exam_subjects"] = keyword
            break
    college_match = re.search(r"([\u4e00-\u9fff]{2,20}学院)", query)
    if college_match:
        filters["college_name"] = college_match.group(1)
    major_match = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{2,40}专业)", query)
    if major_match:
        filters["major_name"] = major_match.group(1).removesuffix("专业")
    return filters


def _extract_scoreline_filters(query: str) -> dict[str, str]:
    filters: dict[str, str] = {}
    year = _extract_year(query)
    if year:
        filters["year"] = year
    for province in ("河南", "河北", "山东", "山西", "安徽", "江苏", "浙江", "湖北", "湖南", "广东", "广西", "江西", "福建", "北京", "天津", "上海", "重庆", "四川", "云南", "贵州"):
        if province in query:
            filters["province"] = province
            break
    for batch in ("本科批", "一本", "二本", "提前批", "艺术批"):
        if batch in query:
            filters["batch"] = batch
            break
    for category in ("理工", "文史", "物理类", "历史类", "物理", "历史"):
        if category in query:
            filters["category"] = category
            break
    major_match = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{2,40}专业)", query)
    if major_match:
        filters["major_name"] = major_match.group(1).removesuffix("专业")
    score_values = re.findall(r"(\d{3,4})", query)
    if "以上" in query and score_values:
        filters["min_score_min"] = score_values[-1]
    elif "以下" in query and score_values:
        filters["min_score_max"] = score_values[-1]
    return filters


def _extract_policy_filters(query: str) -> dict[str, str]:
    filters: dict[str, str] = {}
    year = _extract_year(query)
    if year:
        filters["year"] = year
    if "专业情况汇总表" in query:
        filters["table_topic"] = "专业情况汇总表"
    filters["keyword"] = query[:80]
    return filters


def _extract_year(query: str) -> str:
    matched = re.search(r"(20\d{2})", query)
    return matched.group(1) if matched else ""
