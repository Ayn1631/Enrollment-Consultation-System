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


def test_route_structured_query_cleans_major_name_prefixes():
    major_decision = route_structured_query("2025年应用化学专业学费和选考科目是什么？")
    score_decision = route_structured_query("2025年河南本科批自动化专业最低分和位次是多少？")

    assert major_decision.filters["major_name"] == "应用化学"
    assert score_decision.filters["major_name"] == "自动化"


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


def test_render_payload_text_returns_full_record_text():
    toolset = StructuredAdmissionsToolset(Settings())
    payload = StructuredToolPayload(
        tool_name="major_catalog_lookup",
        matched_fields=[],
        route_reason="专业目录类结构化查询",
        records=[
            {
                "source_file": "2025年招生专业详情.xlsx",
                "source_doc": "2025年招生专业详情.xlsx",
                "major_code": "080801",
                "major_name": "自动化",
                "duration": "四年",
                "tuition": "5500",
                "exam_subjects": "物理+化学",
                "degree_type": "工学",
                "college_name": "自动化与电气工程学院",
                "evidence_text": "专业名称：自动化；学费（元）：5500；所在院系：自动化与电气工程学院；选考科目：物理+化学",
            }
        ],
    )

    rendered = toolset.render_payload_text(payload, max_chars=500, max_records=1)

    assert "以下为 xlsx 原表格式输出" in rendered
    assert "专业代码\t专业名称\t学制\t学费（元）\t选考科目\t学位授予门类\t所在院系" in rendered
    assert "080801\t自动化\t四年\t5500\t物理+化学\t工学\t自动化与电气工程学院" in rendered


def test_filter_rows_prefers_exact_major_match():
    toolset = StructuredAdmissionsToolset(Settings())
    rows = [
        {
            "academic_year": "2025",
            "major_name": "应用化学",
            "tuition": "5000",
            "exam_subjects": "物理+化学",
            "college_name": "材料电子与储能学院",
            "source_row_no": "2",
        },
        {
            "academic_year": "2025",
            "major_name": "材料成型及控制工程",
            "tuition": "5000",
            "exam_subjects": "物理+化学",
            "college_name": "材料电子与储能学院",
            "source_row_no": "3",
        },
    ]

    matched = toolset._filter_rows(
        rows,
        raw_query="2025年材料成型及控制工程专业学费和选考科目是什么？",
        filters={"academic_year": "2025", "major_name": "材料成型及控制工程"},
        limit=2,
    )

    assert matched[0]["major_name"] == "材料成型及控制工程"


def test_major_catalog_lookup_fallbacks_when_repository_returns_empty(monkeypatch):
    toolset = StructuredAdmissionsToolset(Settings())
    monkeypatch.setattr(toolset.repository, "search_major_catalog", lambda **_: [])
    monkeypatch.setattr(
        toolset,
        "_search_major_catalog_fallback",
        lambda **_: [{"major_name": "软件工程", "source_file": "2025年招生专业详情.xlsx", "source_row_no": "1"}],
    )

    payload = toolset.major_catalog_lookup(
        raw_query="2025年软件工程专业学费和选考科目是什么？",
        filters={"academic_year": "2025", "major_name": "软件工程"},
    )

    assert payload.records[0]["major_name"] == "软件工程"


def test_choose_best_payload_prefers_more_relevant_result():
    toolset = StructuredAdmissionsToolset(Settings())
    major_payload = StructuredToolPayload(
        tool_name="major_catalog_lookup",
        matched_fields=[],
        route_reason="专业目录类结构化查询",
        records=[
            {
                "major_name": "自动化",
                "college_name": "自动化与电气工程学院",
                "evidence_text": "专业名称：自动化；学费（元）：5500；所在院系：自动化与电气工程学院",
                "_retrieval_score": 92.0,
            }
        ],
    )
    scoreline_payload = StructuredToolPayload(
        tool_name="scoreline_lookup",
        matched_fields=[],
        route_reason="录取分数/位次类结构化查询",
        records=[
            {
                "major_name": "自动化",
                "province": "河南",
                "year": "2025",
                "evidence_text": "2025年河南本科批自动化最低分 531",
                "_retrieval_score": 48.0,
            }
        ],
    )

    best = toolset.choose_best_payload([scoreline_payload, major_payload], raw_query="自动化专业学费是多少")

    assert best is not None
    assert best.tool_name == "major_catalog_lookup"


def test_major_catalog_fulltext_should_return_all_repository_rows(monkeypatch):
    toolset = StructuredAdmissionsToolset(Settings())
    captured: dict[str, object] = {}

    def _fake_search_major_catalog(*, raw_query: str, filters: dict[str, str], limit):
        captured["raw_query"] = raw_query
        captured["filters"] = filters
        captured["limit"] = limit
        return [
            {"major_name": "自动化", "source_file": "a.xlsx", "evidence_text": "自动化全文"},
            {"major_name": "机器人工程", "source_file": "a.xlsx", "evidence_text": "机器人工程全文"},
        ]

    monkeypatch.setattr(toolset.repository, "search_major_catalog", _fake_search_major_catalog)

    payload = toolset.major_catalog_fulltext()

    assert captured == {"raw_query": "", "filters": {}, "limit": None}
    assert len(payload.records) == 2
    assert payload.records[0]["major_name"] == "自动化"


def test_render_policy_payload_text_should_pivot_to_table_rows():
    toolset = StructuredAdmissionsToolset(Settings())
    payload = StructuredToolPayload(
        tool_name="policy_table_lookup",
        matched_fields=[],
        route_reason="政策附表类结构化全量文本返回",
        records=[
            {
                "source_file": "附表.xlsx",
                "table_topic": "专业情况汇总表",
                "source_row_no": "26",
                "field_name": "major_name",
                "field_value": "英语",
            },
            {
                "source_file": "附表.xlsx",
                "table_topic": "专业情况汇总表",
                "source_row_no": "26",
                "field_name": "tuition",
                "field_value": "4400",
            },
            {
                "source_file": "附表.xlsx",
                "table_topic": "专业情况汇总表",
                "source_row_no": "26",
                "field_name": "evidence_text",
                "field_value": "专业代码：050201；专业名称：英语",
            },
        ],
    )

    rendered = toolset.render_payload_text(payload, max_records=10)

    assert "命中记录数：1" in rendered
    assert "专业名称\t学费（元）" in rendered
    assert "英语\t4400" in rendered
    assert "来源文件" not in rendered
    assert "evidence_text" not in rendered


def test_render_policy_payload_text_should_preserve_numeric_row_order():
    toolset = StructuredAdmissionsToolset(Settings())
    payload = StructuredToolPayload(
        tool_name="policy_table_lookup",
        matched_fields=[],
        route_reason="政策附表类结构化全量文本返回",
        records=[
            {"source_file": "附表.xlsx", "table_topic": "major_catalog", "source_row_no": "10", "field_name": "major_code", "field_value": "080203"},
            {"source_file": "附表.xlsx", "table_topic": "major_catalog", "source_row_no": "10", "field_name": "major_name", "field_value": "材料成型及控制工程"},
            {"source_file": "附表.xlsx", "table_topic": "major_catalog", "source_row_no": "2", "field_name": "major_code", "field_value": "070302"},
            {"source_file": "附表.xlsx", "table_topic": "major_catalog", "source_row_no": "2", "field_name": "major_name", "field_value": "应用化学"},
        ],
    )

    rendered = toolset.render_payload_text(payload, max_records=10)
    lines = [line for line in rendered.splitlines() if line.strip()]

    assert "070302\t应用化学" in lines[4]
    assert "080203\t材料成型及控制工程" in lines[5]
