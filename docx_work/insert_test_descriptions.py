from pathlib import Path
import html
import sys

sys.path.insert(0, r"C:\Users\20753\.codex\skills\docx")

from scripts.document import Document, DocxXMLEditor
from ooxml.scripts.pack import pack_document


UNPACKED = Path("docx_work/test_desc_unpacked")
MODIFIED = Path("docx_work/test_desc_modified_unpacked")
OUTPUT = Path("曾令洁-毕业论文-基于RAG与工作流的中原工学院招生咨询系统-测试部分描述增强.docx")
RSID = "D8DED8BF"


def tracked_paragraph(style_source, text):
    ppr_nodes = style_source.getElementsByTagName("w:pPr")
    ppr = ppr_nodes[0].toxml() if ppr_nodes else ""
    body = f"<w:p>{ppr}<w:r><w:t>{html.escape(text, quote=False)}</w:t></w:r></w:p>"
    return DocxXMLEditor.suggest_paragraph(body)


def insert_paragraphs(editor, target, style_source, texts):
    cursor = target
    for text in texts:
        inserted = editor.insert_after(cursor, tracked_paragraph(style_source, text))
        cursor = inserted[-1]


doc = Document(UNPACKED, rsid=RSID, track_revisions=True, author="Codex", initials="CDX")
editor = doc["word/document.xml"]

env_style = editor.get_node(tag="w:p", contains="本项目测试并不只停留在页面演示层面")
feature_style = editor.get_node(tag="w:p", contains="为了测试系统基本可行性")
rag_style = editor.get_node(tag="w:p", contains="RAG的三指标测试结果如表6-3所示")
perf_style = editor.get_node(tag="w:p", contains="根据表 6-4 可知")
discussion_style = editor.get_node(tag="w:p", contains="系统已经建立了较为完整的降级保障策略")

insert_paragraphs(
    editor,
    editor.get_node(tag="w:tbl", contains="FAISS+BM25+Exact Match"),
    env_style,
    [
        "从测试组织方式看，第六章各项验证均基于同一套后端服务、前端页面和知识库索引展开，功能回放、RAG评测和并发压测共享主要业务链路。这样的安排可以减少环境差异对结果的影响，也便于将接口行为、检索指标和性能日志对应到同一套实现上。",
    ],
)

insert_paragraphs(
    editor,
    editor.get_node(tag="w:tbl", contains="索引重建与上下文压缩"),
    feature_style,
    [
        "功能测试部分重点观察已有接口和页面能力是否能够串联成完整问答流程。普通聊天和SSE流式聊天验证了网关入口与前端事件接收的连通性，模式切换验证了chat、rag、agent三类处理路径的可用性，健康检查、索引重建和上下文压缩则用于确认系统在调试和维护阶段具备基本自检能力。",
        "从结果看，表6-2中的验证项目均为系统实现章节已经给出的核心功能，没有额外引入独立测试场景。其意义在于证明前端交互、后端网关、RAG索引和运维接口之间能够形成闭环，而不仅是单个接口能够单独返回结果。",
    ],
)

insert_paragraphs(
    editor,
    editor.get_node(tag="w:tbl", contains="0.6976"),
    rag_style,
    [
        "进一步分析三项指标可以发现，MRR@5较高说明多数样本的首个相关证据排序靠前，用户在常规问答中较容易获得可引用材料；nDCG@5达到0.8078，说明前5条候选结果并非随机堆叠，而是具有一定相关性分布。Recall@5相对偏低则表明部分问题虽然能够命中关键证据，但同一问题所需的补充证据覆盖不足，尤其在跨专业、跨年份或需要组合多个字段的问题中更容易暴露。",
        "因此，RAG专项评测的结论不能简单理解为检索链路已经完全稳定，而应理解为排序质量先于召回完整性达到可用水平。后续优化应优先围绕语料切分、字段归一、查询改写触发条件和召回融合权重展开，而不是盲目增加模型生成约束。",
    ],
)

insert_paragraphs(
    editor,
    perf_style,
    perf_style,
    [
        "从容错角度看，高并发场景下系统没有把问题主要表现为接口不可用，而是表现为首token等待时间和完整响应耗时增加。这说明熔断、降级和质量闸门能够在一定程度上维持主链路可运行，但当前同步执行的检索准备步骤仍会成为延迟集中点。对于招生咨询场景，响应稳定性和依据可靠性比极限吞吐更重要，因此本文暂时保留质量优先策略，但后续需要通过缓存、异步预取和检索阶段裁剪来降低等待时间。",
    ],
)

insert_paragraphs(
    editor,
    discussion_style,
    discussion_style,
    [
        "总体而言，第六章测试结果与系统设计目标基本一致：功能测试验证可用性，RAG专项评测揭示检索排序与召回差异，性能数据说明主链路在压力下的主要瓶颈，结果讨论则把这些现象回收到系统改进方向中。上述分析均基于已有测试结果展开，未改变测试范围，只补充了对结果含义和工程取舍的解释。",
    ],
)

doc.save(MODIFIED)
pack_document(MODIFIED, OUTPUT, validate=False)
print(OUTPUT)
