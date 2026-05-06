from __future__ import annotations

from app.admissions_kb.router import route_structured_query
from app.admissions_kb.tools import StructuredAdmissionsToolset, StructuredToolPayload
from app.config import Settings


def test_route_structured_query_prefers_scoreline():
    decision = route_structured_query("2025年河南本科批自动化专业最低分和位次是多少？")

    assert decision.tool_name == "scoreline_lookup"
    assert decision.filters["year"] == "2025"
    assert decision.filters["province"] == "河南"


def test_route_structured_query_prefers_major_catalog():
    decision = route_structured_query("自动化专业学费和选考科目是什么？")

    assert decision.tool_name == "major_catalog_lookup"


def test_route_structured_query_prefers_policy_table():
    decision = route_structured_query("招生章程里的专业情况汇总表怎么写的？")

    assert decision.tool_name == "policy_table_lookup"


def test_toolset_to_rag_response_formats_records():
    toolset = StructuredAdmissionsToolset(Settings())
    payload = StructuredToolPayload(
        tool_name="major_catalog_lookup",
        matched_fields=["major_name", "academic_year"],
        route_reason="专业目录类结构化查询",
        records=[
            {
                "source_file": "2025年招生专业详情.xlsx",
                "major_name": "自动化",
                "college_name": "自动化与电气工程学院",
                "evidence_text": "专业名称：自动化；学费（元）：5500；所在院系：自动化与电气工程学院",
            }
        ],
    )

    response = toolset.to_rag_response(payload=payload, trace_id="structured-major")

    assert response is not None
    assert response.status == "ok"
    assert response.sources[0].title == "自动化 - 自动化与电气工程学院"
    assert "structured:major_catalog_lookup" in response.context_blocks[0]
