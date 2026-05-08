from app.config import Settings
from app.admissions_kb.tools import StructuredAdmissionsToolset

settings = Settings()
toolset = StructuredAdmissionsToolset(settings)

payloads = [
    toolset.major_catalog_lookup(
        raw_query="自动化专业学费、选考科目、所属学院是什么？",
        filters={},
    ),
    toolset.scoreline_lookup(
        raw_query="2025年河南本科批自动化专业最低分和最低位次是多少？",
        filters={},
    ),
    toolset.policy_table_lookup(
        raw_query="本科招生章程里的专业情况汇总表有哪些内容？",
        filters={},
    ),
]

for idx, payload in enumerate(payloads, start=1):
    print("=" * 100)
    print(f"工具 {idx}: {payload.tool_name}")
    print(toolset.render_payload_text(payload))
    print()