from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable

from app.admissions_kb.tools import StructuredAdmissionsToolset
from app.models import AgentStepEvent, AgentStrategy, ChatRequest, ChatSource, FeatureFlag, SessionResult
from app.services.ai_stack import (
    McpToolRuntime,
    build_langchain_chat_model,
    build_langchain_mcp_runtime,
    load_mcp_server_configs,
)
from app.services.agent_types import (
    PlanStep,
    StepExecutionResult,
    StepReviewResult,
    SubproblemState,
)

if TYPE_CHECKING:
    from app.services.gateway import GatewayOrchestrator


StepSink = Callable[[AgentStepEvent], None]


class AgentRuntime:
    def __init__(self, gateway: GatewayOrchestrator):
        self.gateway = gateway
        self.logger = logging.getLogger(__name__)
        self._mcp_runtime_cache: dict[str, McpToolRuntime] = {}
        self._mcp_runtime_locks: dict[str, threading.RLock] = {}
        self._mcp_runtime_guard = threading.RLock()
        self._rag_document_catalog_text: str | None = None
        self._tool_trace_log_lock = threading.RLock()

    @property
    def deps(self):
        return self.gateway.deps

    def new_trace_id(self) -> str:
        return uuid.uuid4().hex

    def get_last_user(self, request: ChatRequest) -> str:
        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "").strip()
        return last_user or "请介绍中原工学院招生政策要点。"

    def emit_step(
        self,
        *,
        events: list[AgentStepEvent],
        sink: StepSink | None,
        strategy: AgentStrategy,
        node: str,
        title: str,
        status: str,
        message: str | None = None,
        subproblem_id: str | None = None,
        plan_step_index: int | None = None,
        attempt: int | None = None,
    ) -> AgentStepEvent:
        event = AgentStepEvent(
            id=uuid.uuid4().hex,
            node=node,
            title=title,
            status=status,  # type: ignore[arg-type]
            message=message,
            subproblem_id=subproblem_id,
            plan_step_index=plan_step_index,
            attempt=attempt,
            strategy=strategy,
            timestamp=datetime.utcnow().isoformat(),
        )
        events.append(event)
        if sink is not None:
            sink(event)
        return event

    def _serialize_tool_payload(self, payload: Any) -> str:
        if payload is None:
            return "无"
        if isinstance(payload, str):
            return payload.strip() or "无"
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            return str(payload).strip() or "无"

    def _truncate_tool_trace_text(self, text: str, *, max_chars: int) -> str:
        normalized = (text or "").strip()
        if len(normalized) <= max_chars:
            return normalized or "无"
        return normalized[: max_chars - 16].rstrip() + "\n...（已截断）"

    def _extract_message_tool_calls(self, message: Any) -> list[dict[str, Any]]:
        raw_tool_calls = getattr(message, "tool_calls", None)
        if not raw_tool_calls:
            additional_kwargs = getattr(message, "additional_kwargs", None)
            if isinstance(additional_kwargs, dict):
                raw_tool_calls = additional_kwargs.get("tool_calls")
        normalized_calls: list[dict[str, Any]] = []
        for item in raw_tool_calls or []:
            if not isinstance(item, dict):
                continue
            function_block = item.get("function") if isinstance(item.get("function"), dict) else {}
            args = item.get("args")
            if args is None:
                args = item.get("arguments")
            if args is None:
                args = function_block.get("arguments")
            if isinstance(args, str):
                stripped = args.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    try:
                        args = json.loads(stripped)
                    except Exception:
                        args = stripped
            normalized_calls.append(
                {
                    "id": str(item.get("id") or item.get("tool_call_id") or ""),
                    "name": str(item.get("name") or function_block.get("name") or ""),
                    "args": args,
                }
            )
        return normalized_calls

    def _pop_matching_tool_call(
        self,
        pending_tool_calls: list[dict[str, Any]],
        *,
        tool_name: str,
        tool_call_id: str,
    ) -> dict[str, Any] | None:
        if tool_call_id:
            for index, item in enumerate(pending_tool_calls):
                if str(item.get("id") or "") == tool_call_id:
                    return pending_tool_calls.pop(index)
        if tool_name:
            for index, item in enumerate(pending_tool_calls):
                if str(item.get("name") or "") == tool_name:
                    return pending_tool_calls.pop(index)
        return None

    def _build_tool_trace_message(self, *, tool_name: str, tool_args: Any, tool_output: str) -> str:
        serialized_args = self._truncate_tool_trace_text(
            self._serialize_tool_payload(tool_args),
            max_chars=800,
        )
        serialized_output = self._truncate_tool_trace_text(
            self._serialize_tool_payload(tool_output),
            max_chars=1500,
        )
        return (
            f"工具：{tool_name or 'unknown_tool'}\n"
            f"传入：\n{serialized_args}\n\n"
            f"传出：\n{serialized_output}"
        ).strip()

    def _get_tool_trace_log_path(self) -> Path:
        reports_dir = Path(__file__).resolve().parents[2] / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        return reports_dir / "agent_tool_traces.jsonl"

    def _append_tool_trace_record(
        self,
        *,
        trace_id: str,
        session_id: str,
        tool_name: str,
        tool_args: Any,
        tool_output: str,
        subproblem_id: str | None,
        plan_step_index: int | None,
        attempt: int | None,
    ) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "trace_id": trace_id,
            "session_id": session_id,
            "tool_name": tool_name or "unknown_tool",
            "tool_args": tool_args,
            "tool_output": tool_output,
            "subproblem_id": subproblem_id,
            "plan_step_index": plan_step_index,
            "attempt": attempt,
        }
        path = self._get_tool_trace_log_path()
        with self._tool_trace_log_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.logger.info(
            "[agent.tool_trace] trace_id=%s session_id=%s tool=%s step=%s attempt=%s output_chars=%s",
            trace_id,
            session_id,
            tool_name or "unknown_tool",
            plan_step_index,
            attempt,
            len(tool_output or ""),
        )

    def audit_user_input(self, query: str) -> tuple[bool, str, str]:
        return self.gateway._audit_user_input(query)

    def audit_generated_output(self, text: str) -> tuple[bool, str, str]:
        return self.gateway._audit_generated_output(text)

    def route_features(self, query: str, request: ChatRequest):
        return self.gateway._route_features(query, request)

    def dedupe_sources(self, sources: list[ChatSource], limit: int = 5) -> list[ChatSource]:
        return self.gateway._dedupe_chat_sources(sources, limit=limit)

    def build_failure_session(
        self,
        *,
        request: ChatRequest,
        exc: Exception,
        step_events: list[AgentStepEvent] | None = None,
    ) -> SessionResult:
        session = self.gateway._build_agent_failure_session(request=request, exc=exc)
        session.agent_strategy = request.agent_strategy
        session.agent_trace = list(step_events or [])
        return session

    def load_memory_context(self, session_id: str) -> tuple[list[str], str, list[str]]:
        context_blocks: list[str] = []
        notes: list[str] = []
        memory_lines: list[str] = []
        for kind, label, prefix in (
            ("short", "短期记忆", "[memory]"),
            ("long", "长期记忆", "[long-memory]"),
            ("special", "特殊记忆", "[special-memory]"),
        ):
            result = self.deps.container.isolation.execute(
                "memory-service",
                lambda kind=kind: self.deps.services.read_memory(session_id=session_id, kind=kind),
            )
            if not result.ok or result.value is None:
                notes.append(f"{label}读取失败，已忽略。")
                continue
            entries = result.value.entries[:3]
            if not entries:
                continue
            context_blocks.extend([f"{prefix} {item.value}" for item in entries])
            memory_lines.append(f"{label}：")
            memory_lines.extend([f"- {item.key}: {item.value}" for item in entries])
            notes.append(f"{label}已接入上下文。")
        return context_blocks, "\n".join(memory_lines) if memory_lines else "当前没有可用记忆。", notes

    def _get_rag_document_catalog_text(self) -> str:
        cached = self._rag_document_catalog_text
        if cached is not None:
            return cached
        docs_root = Path(__file__).resolve().parents[3] / "docs"
        if not docs_root.exists():
            self._rag_document_catalog_text = "当前未扫描到本地知识库文档清单。"
            return self._rag_document_catalog_text
        names = sorted(
            str(path.relative_to(docs_root)).replace("\\", "/")
            for path in docs_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".txt", ".csv"}
        )
        if not names:
            self._rag_document_catalog_text = "当前未扫描到本地知识库文档清单。"
            return self._rag_document_catalog_text
        self._rag_document_catalog_text = "；".join(names)
        return self._rag_document_catalog_text

    def _has_mcp_servers(self) -> bool:
        servers, _ = load_mcp_server_configs(self.deps.services.settings)
        return bool(servers)

    def rewrite_query(
        self,
        *,
        request: ChatRequest,
        last_user: str,
        memory_text: str,
        strategy: AgentStrategy,
    ) -> str:
        recent_user_context = self._build_recent_user_context(request=request)
        llm = build_langchain_chat_model(
            self.deps.services.settings,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        if llm is None:
            return self._rule_rewrite_query(
                last_user=last_user,
                memory_text=memory_text,
                strategy=strategy,
            )
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except Exception:
            return self._rule_rewrite_query(
                last_user=last_user,
                memory_text=memory_text,
                strategy=strategy,
            )
        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "你是“招生咨询查询改写器”。"
                            "你的任务是把用户当前这句话，改写成适合招生咨询工具链执行的单轮查询。"
                            "你的输出会直接用于后续检索、RAG、技能执行、外部搜索或工具路由，因此改写结果必须清晰、完整、可检索、少歧义。"
                            "你必须先在内部综合理解“执行策略、用户原问题、最近多轮用户上下文、相关记忆”，再给出最终改写结果，但不要输出你的分析过程。"
                            "你只能输出最终改写后的问题文本，不要输出解释、前缀、编号、引号、Markdown、多个候选版本或任何额外内容。"
                            "改写目标如下："
                            "第一，补全省略代词、指代对象、年份、省份、专业、费用项、流程对象等关键缺失信息，使问题变成可以独立理解的单轮查询。"
                            "第二，尽量利用最近多轮用户上下文和相关记忆补全上下文，但只能使用与当前问题直接相关、且可以高置信度承接的信息。"
                            "第三，必须保持用户原意，不得编造新需求，不得扩写成用户没有表达过的目标。"
                            "第四，如果记忆里存在多个可能指代对象且无法确定，就保持原问题核心意图并做最小保守补全，不要瞎猜。"
                            "第五，不要回答问题本身，你的职责只是改写查询，不是提供结论。"
                            "第六，输出必须是适合工具链理解的一条自然语言查询，而不是搜索语法、标签列表或问答摘要。"
                            "第七，当用户问题已经足够清晰时，只做轻量规范化，不要过度改写。"
                            "第八，年份、省份、分数、科类/选科、批次、专业名称、学校名称等如果已经在最近多轮用户上下文中明确给出，且当前问题没有明确推翻，就应在改写结果中完整保留，不得遗漏。"
                            "记忆使用规则："
                            "只有在原问题出现“这个、那个、它、那、这、该专业、这个学费、今年、这类情况”等省略表达时，才优先用记忆补全。"
                            "如果记忆与原问题无关，忽略记忆。"
                            "如果记忆能明确定位用户当前追问对象，应直接替换省略指代，使改写结果可独立成立。"
                            "如果记忆只能提供弱线索，宁可保守，也不要硬补。"
                            "策略适配规则："
                            "当执行策略是 speed 时，优先生成更短、更直接、最小必要补全的查询。"
                            "当执行策略是 quality 时，在不改变原意的前提下，生成信息更完整、歧义更少、检索更稳定的查询。"
                            "下面是示例："
                            "示例1："
                            "原问题：这个专业学费多少？"
                            "相关记忆：用户前文持续咨询软件工程专业。"
                            "输出：中原工学院软件工程专业学费多少？"
                            "示例2："
                            "原问题：那河南理科呢？"
                            "相关记忆：用户前文在问中原工学院软件工程专业在河南理科的录取情况。"
                            "输出：中原工学院软件工程专业在河南理科的录取情况如何？"
                            "示例3："
                            "原问题：今年报名流程是啥？"
                            "相关记忆：用户正在咨询中原工学院专升本报考。"
                            "输出：中原工学院今年专升本报名流程是什么？"
                            "示例4："
                            "原问题：住宿费呢"
                            "相关记忆：用户前文已经明确在问 2026 年本科新生收费标准。"
                            "输出：中原工学院 2026 年本科新生住宿费是多少？"
                            "示例5："
                            "原问题：这个学校怎么样"
                            "相关记忆：无明确相关记忆。"
                            "输出：中原工学院怎么样？"
                        )
                    ),
                    HumanMessage(
                        content=(
                            "请执行一次招生咨询查询改写任务。\n\n"
                            f"执行策略：{strategy}\n"
                            f"用户原问题：{last_user}\n\n"
                            f"最近多轮用户上下文：\n{recent_user_context}\n\n"
                            f"相关记忆：\n{memory_text}\n\n"
                            "输出要求：\n"
                            "1. 只输出一条改写后的查询文本。\n"
                            "2. 优先补全当前问题中的省略指代、时间范围、地区、专业、费用项、流程对象等关键缺失信息。\n"
                            "3. 可以根据最近多轮用户上下文与相关记忆替换“这个、那个、它、那、今年、该专业、这个费用”等模糊表达，但前提是这些上下文与当前问题直接相关且指向明确。\n"
                            "4. 不要编造用户没有提出的新目标，不要把一个问题扩成多个问题。\n"
                            "5. 不要输出解释，不要输出分析过程，不要输出多个候选版本。\n"
                            "6. 如果原问题已经足够清晰，就做轻量规范化后直接输出。\n"
                            "7. 如果记忆不足以安全替换代词，就保守改写，不要乱猜。\n"
                            "8. 必须完整保留最近多轮用户上下文中已经明确给出的条件，尤其是年份、省份、分数、科类/选科、批次、专业、学校。\n"
                            "9. 如果当前问题要求“参考往年情况给推荐”，则改写结果中必须同时保留当前目标年份和参考年份。"
                        )
                    ),
                ]
            )
            content = str(getattr(response, "content", "") or "").strip()
            return content or self._rule_rewrite_query(
                last_user=last_user,
                memory_text=memory_text,
                strategy=strategy,
            )
        except Exception:
            return self._rule_rewrite_query(
                last_user=last_user,
                memory_text=memory_text,
                strategy=strategy,
            )

    def split_query(
        self,
        query: str,
        strategy: AgentStrategy,
        request: ChatRequest | None = None,
        memory_text: str = "",
    ) -> list[str]:
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            return []
        max_subproblems = 5 if strategy == "speed" else 20
        llm = build_langchain_chat_model(
            self.deps.services.settings,
            model=request.model if request is not None else None,
            temperature=request.temperature if request is not None else None,
            top_p=request.top_p if request is not None else None,
        )
        if llm is None:
            return self._rule_split_query(normalized, max_subproblems=max_subproblems)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except Exception:
            return self._rule_split_query(normalized, max_subproblems=max_subproblems)
        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            '''
你是“招生咨询系统子问题拆分器”。

你的任务是把用户的招生咨询问题，拆分成一组“可独立求解、合起来又能覆盖原始意图”的子问题列表，用于后续检索、工具调用、流程分析、政策核验和答案综合。

你的工作不是回答问题，而是为后续工具链生成高质量的求解子任务。
你必须先在内部充分理解原问题的真实目标、约束条件、隐含前置依赖、可能需要深化的维度，再输出最终结果。
但不要输出你的思考过程，只输出最终 JSON 数组。

【一、领域约束】
当前任务领域是“招生咨询系统”。
你处理的问题通常涉及：
- 招生政策
- 报考条件
- 专业信息
- 学费与住宿费
- 奖助学金与资助
- 分数线、位次、录取规则
- 报名、报到、转专业、专升本等流程
- 学校、学院、专业对比
- 时间敏感的年度招生信息

因此，你拆分出来的子问题必须适合招生咨询场景，不要泛化成空洞问题，也不要偏离用户在招生上的真实关注点。

【二、核心目标】
拆分后的子问题必须满足以下目标：
1. 每个子问题都能被单独检索、单独分析、单独求解。
2. 所有子问题合起来，能够覆盖用户原问题的主要意图。
3. 不能编造用户没有提出的新问题。
4. 不能把一个本来只适合整体回答的问题硬拆碎。
5. 对于复杂问题，要识别“前置问题”和“深化问题”。

【三、前置问题与深化问题】
你必须理解这两个概念：

1. 前置问题
指为了正确回答主问题，必须先确认的基础问题、限定条件或判断依据。
例如：
- 用户问“我能不能报这个专业”，前置问题可能是该专业的报考条件、选科要求、分数要求。
- 用户问“学费贵不贵”，前置问题可能是具体专业、学历层次、年份、收费标准。

2. 深化问题
指在主问题已经基本成立后，进一步展开的关键细分问题。
例如：
- 用户问“这个专业值不值得报”，深化问题可能是就业、课程设置、学费、录取难度。
- 用户问“怎么申请助学金”，深化问题可能是申请条件、材料、流程、时间节点。

拆分时，你需要先判断：
- 原问题是否包含隐含前置依赖
- 原问题是否天然包含多个需要展开的深化维度

如果有，就拆出来；
如果没有，就不要为了显得聪明硬拆。

【四、拆分原则】
你必须严格遵守以下原则：

1. 保持原意
所有子问题都必须忠实于用户原问题，不得扩展出用户没有表达或明显暗示的新目标。

2. 保留硬约束
如果原问题里有年份、省份、专业、批次、分数、金额、身份、对象、条件、否定词等约束，拆分后必须尽可能保留在相关子问题中。

3. 子问题要“独立可求解”
每个子问题都应当是后续系统可以单独处理的一条清晰任务，不要写成半截话、代词指代不清的话、或必须依赖别的子问题才能理解的话。

4. 先前置，后深化
如果需要拆出前置问题和深化问题，优先把前置问题放前面，再放深化问题，保持求解顺序合理。

5. 去掉伪拆分
不要把同一句话换个说法当成两个子问题。
不要把一个问题机械拆成字面近义重复项。

6. 控制数量
speed 策略时拆分更保守，只拆最必要的部分。
quality 策略时可以拆得更细，但仍然必须克制，不要超过最大上限，不要为了凑数量而拆。

7. 单问题不乱拆
如果原问题本身就是单一问题，或者拆开反而损失上下文，就返回只包含一个元素的数组。

【五、适合拆分的典型情况】
以下情况通常适合拆分：
- 明确包含多个并列意图
- 一个主问题依赖若干前置判断
- 一个抽象问题需要拆成多个评价维度
- 一个流程问题天然包含条件、材料、步骤、时间等多个子面向
- 一个比较问题天然包含多个对比维度

【六、不适合拆分的典型情况】
以下情况通常不适合硬拆：
- 单一事实问答
- 明确的单点费用问题
- 单一联系方式问题
- 单一时间节点问题
- 非常短且语义单一的问题
- 拆开后会造成大量重复、歧义或信息割裂的问题

【七、输出要求】
你必须严格遵守以下输出规则：
1. 只输出 JSON 数组。
2. 数组元素必须是字符串。
3. 不要输出 Markdown。
4. 不要输出解释、分析、标题、前后缀、代码块。
5. 不要输出对象数组，不要输出带字段的结构。
6. 每个字符串都必须是一个完整、自然、可独立求解的子问题。
7. 如果只需要一个子问题，就返回只含一个元素的数组。

【八、质量标准】
在输出前，默默检查：
- 有没有偏离用户原意
- 有没有凭空新增需求
- 有没有漏掉关键约束
- 有没有把前置问题和深化问题顺序搞反
- 有没有重复子问题
- 有没有拆得过细或过粗
- 每个子问题是否都能单独求解

【九、示例】

示例1：单一问题，不应硬拆
用户问题：
“中原工学院招生办电话是多少？”

输出：
["中原工学院招生办电话是多少？"]

示例2：并列意图，直接拆分
用户问题：
“中原工学院软件工程专业学费多少，住宿费多少，奖学金好申请吗？”

输出：
["中原工学院软件工程专业学费是多少？","中原工学院住宿费标准是多少？","中原工学院奖学金申请难度和基本条件如何？"]

示例3：包含前置问题和深化问题
用户问题：
“我这个分数报中原工学院计算机类希望大吗？”

输出：
["中原工学院计算机类专业近年录取分数或位次要求如何？","我的分数是否达到报考中原工学院计算机类专业的基本范围？","如果达到基本范围，报考中原工学院计算机类专业的录取把握如何？"]

示例4：流程问题，需要拆成前置+步骤
用户问题：
“中原工学院助学金怎么申请？”

输出：
["中原工学院助学金申请的基本条件是什么？","中原工学院助学金申请需要准备哪些材料？","中原工学院助学金申请流程和时间节点是什么？"]

示例5：抽象评价问题，需要拆出深化维度
用户问题：
“中原工学院软件工程专业值得报吗？”

输出：
["中原工学院软件工程专业的培养内容和课程设置如何？","中原工学院软件工程专业的就业方向和就业情况如何？","中原工学院软件工程专业的录取难度和学费情况如何？"]

示例6：比较问题，拆出对比维度
用户问题：
“中原工学院和河南工程学院哪个更适合学机械？”

输出：
["中原工学院机械相关专业情况如何？","河南工程学院机械相关专业情况如何？","中原工学院和河南工程学院在机械专业培养、就业和录取难度上如何对比？"]

示例7：quality 策略下可适当细化
用户问题：
“专升本报名需要注意什么？”

输出：
["中原工学院专升本报名的基本条件是什么？","中原工学院专升本报名需要准备哪些材料？","中原工学院专升本报名流程是什么？","中原工学院专升本报名有哪些常见限制或注意事项？"]

示例8：不要编造不存在的需求
用户问题：
“学费贵吗？”

错误拆分：
["中原工学院学费是多少？","中原工学院宿舍怎么样？","中原工学院就业率高吗？"]

正确输出：
["中原工学院相关专业学费是多少？","中原工学院学费在同类院校中处于什么水平？"]

现在开始执行任务时，你必须先准确判断这个问题是否值得拆、该拆成几步、哪些是前置问题、哪些是深化问题，然后只输出最终 JSON 数组。

                            '''
                            )
                    ),
                    HumanMessage(
                        content=(
                            f'''
执行策略：{strategy}
最大子问题数：{max_subproblems}
原问题：{normalized}
相关记忆：
{memory_text or "当前没有可用记忆。"}

请根据上面的约束拆分子问题。
要求：
1. 只返回 JSON 数组。
2. 每个元素都是可独立求解的字符串子问题。
3. 如有必要，优先输出前置问题，再输出深化问题。
4. 可以利用相关记忆补全省略指代、科类、省份、年份、专业等上下文，但不能编造记忆中没有的信息。
5. 不要编造新问题，不要遗漏关键约束。
6. 如果原问题不适合拆分，就返回单元素数组。

示例格式：
["子问题1", "子问题2"]

                            '''
                        )
                    ),
                ]
            )
            raw_content = str(getattr(response, "content", "") or "").strip()
            parsed = json.loads(raw_content)
            if not isinstance(parsed, list):
                return self._rule_split_query(normalized, max_subproblems=max_subproblems)
            parts = [
                re.sub(r"\s+", " ", str(item)).strip(" \t\r\n，。；;？?！!")
                for item in parsed
                if str(item).strip()
            ]
            parts = [item for item in parts if item]
            if not parts:
                return self._rule_split_query(normalized, max_subproblems=max_subproblems)
            return parts[:max_subproblems]
        except Exception:
            return self._rule_split_query(normalized, max_subproblems=max_subproblems)

    def build_plan(
        self,
        *,
        query: str,
        effective_features: list[FeatureFlag],
        route_label: str,
        request: ChatRequest,
        strategy: AgentStrategy,
        memory_text: str = "",
    ) -> list[PlanStep]:
        rag_document_catalog = self._get_rag_document_catalog_text()
        capability_lines = [
            f"- 本地知识库检索（RAG）：{'可用' if 'rag' in effective_features else '不可用'}",
            f"- MCP 外部工具：{'可用' if self._has_mcp_servers() else '不可用'}",
            f"- 历史记忆上下文：{'可用' if memory_text.strip() and memory_text.strip() != '当前没有可用记忆。' else '可用但可能为空'}",
        ]

        llm = build_langchain_chat_model(
            self.deps.services.settings,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        if llm is None:
            return self._rule_build_plan(query)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except Exception:
            return self._rule_build_plan(query)
        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "你是“招生咨询系统子问题执行计划制定器”。"
                            "你的任务是针对当前子问题，生成一组按顺序执行的计划目标，供后续 ReAct Agent 逐步执行。"
                            "你必须先充分理解子问题本身、执行策略、问题路由，以及当前工具能力边界，再输出最终计划。"
                            "你可以在内部深入思考，但绝对不要输出思考过程，只输出最终 JSON 数组。"
                            "领域固定为招生咨询系统，因此你的计划必须围绕招生政策、报考条件、流程、费用、录取、专业、资助、联系方式、时间敏感信息等场景设计，不能写成空泛的通用计划。"
                            "你必须遵守以下规则："
                            "1. 同一子问题内的计划步骤是顺序执行的，不同子问题之间才会并发执行，因此当前计划必须考虑前后依赖关系。"
                            "2. 每个计划节点后续都会交给 ReAct Agent 单独执行，所以每一步都必须目标单一、边界清晰、可独立执行。"
                            "3. 优先安排证据收集，再安排证据补强，最后才安排结论综合。"
                            "4. 如果某个结论需要前置证据支撑，就必须把前置证据步骤排在前面。"
                            "5. 计划只负责描述每一步“要完成什么目标”，不能指定必须使用哪一个工具、哪一种能力或哪一类功能。"
                            "6. 后续执行层会自己决定是否调用本地知识库、MCP 外部工具或其他可用工具，所以你的计划里不要写“必须用 RAG”“必须用 MCP”“必须调用某个工具名”之类限制。"
                            "7. 最后一步必须是综合结论类目标，也就是基于前面已获得的证据形成当前子问题答案。"
                            "8. 如果工具能力有限，就生成最小可行计划，不要为了凑复杂度乱加步骤。"
                            "9. 如果同一类证据一步就能拿到，不要机械重复安排多个同类步骤。"
                            "10. 每一步都必须是一条完整自然语言目标，明确说明该步骤要完成什么、关注什么证据、产出什么类型的结果。"
                            "11. 每一步都应体现招生场景中的关键约束，例如年份、省份、专业、费用项、流程条件、来源可靠性、证据充分性和不确定性处理。"
                            "12. 计划应优先服务于“拿证据并形成可核验结论”，而不是堆步骤。"
                            "13. 当子问题涉及最新公告、当年或近年分省录取、位次、分数线、招生计划、公开网页核验、开放网页事实查询或本地资料明显不足时，计划目标中应自然体现“补强或核验公开信息”的需要，但不要把它写成工具指令。"
                            "14. 所谓“开放网页事实查询”，包括但不限于人物身份、现任职务、院系领导、部门负责人、联系电话、邮箱、办公地点、学院主页栏目内容等这类本地知识库未必完整覆盖、但公开网页可能存在明确答案的问题。"
                            "15. 如果本地知识库文档清单里看不出明显覆盖该事实，或者该事实明显依赖学院/部门页面而不是通用招生资料，就应在计划目标中体现“补充公开证据或核验事实”。"
                            "你需要特别理解“前置步骤”和“深化步骤”："
                            "前置步骤是为了回答当前子问题，先确认必要事实、检索必要资料、核验基础证据；"
                            "深化步骤是在已有证据后进一步补强，例如增加外部工具验证、补充公开信息或进行引用校验；"
                            "综合步骤则负责在已有结果上收束当前子问题结论。"
                            "你必须让计划体现合理顺序：先前置，再深化，最后综合。"
                            "输出格式要求："
                            "只输出 JSON 数组。"
                            "数组中的每个元素都必须是字符串，每个字符串就是一个计划步骤目标。"
                            "不要输出 Markdown、解释、前后缀、注释、代码块或额外字段。"
                            "下面是示例："
                            "示例1："
                            "子问题：中原工学院软件工程专业学费是多少？"
                            "输出："
                            "[\"收集中原工学院软件工程专业学费直接相关的可靠证据，重点确认年份、专业名称、收费标准和适用范围。\",\"基于已获得的收费证据给出学费结论，并说明适用范围及不确定性。\"]"
                            "示例2："
                            "子问题：中原工学院今年专升本报名流程是什么？"
                            "输出："
                            "[\"先确认与中原工学院今年专升本报名流程相关的基础证据，重点梳理报名条件、材料、时间节点和流程步骤。\",\"如果基础证据仍不完整或时效性不足，再补强并核验公开信息，确认最新流程是否有更新。\",\"基于前面证据整理当前子问题答案，按流程顺序总结，并明确哪些信息已确认、哪些仍需保守处理。\"]"
                            "示例3："
                            "子问题：河南理科考生报考中原工学院计算机类专业录取把握如何？"
                            "输出："
                            "[\"先确认河南理科考生报考中原工学院计算机类专业所需的录取依据，重点收集年份、省份、科类、专业范围、分数或位次等证据。\",\"如果已有依据不足以支持判断，再补强并核验近年的录取参考信息或公开规则，以提高判断可靠性。\",\"结合已取得的录取证据对当前子问题给出保守判断，明确依据、适用范围和不确定性。\"]"
                            "示例4："
                            "子问题：中原工学院人工智能学院的院长是谁？"
                            "输出："
                            "[\"先确认本地资料中是否已有人工智能学院领导或院系页面相关证据，判断现有证据能否直接回答该人物身份问题。\",\"如果本地资料仍不明确，再补充并核验学院领导栏目、学院主页或其他公开网页中的明确信息。\",\"综合本地与公开证据给出院长姓名；若仍不能确认，明确缺口在哪里，不得把未知说成已知。\"]"
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"执行策略：{strategy}\n"
                            f"问题路由：{route_label}\n"
                            f"子问题：{query}\n"
                            f"相关记忆：\n{memory_text or '当前没有可用记忆。'}\n"
                            f"本地知识库文档清单：\n{rag_document_catalog}\n"
                            "当前可用工具能力：\n"
                            f"{chr(10).join(capability_lines)}\n\n"
                            "请为这个子问题生成顺序执行计划。\n"
                            "要求：\n"
                            "1. 只返回 JSON 数组。\n"
                            "2. 每个元素都必须是字符串，每个字符串都是一步计划目标。\n"
                            "3. 计划步骤数量尽量精简但必须有效，避免重复和空步骤。\n"
                            "4. 必须先考虑前置证据步骤，再考虑证据补强或公开信息核验，最后以综合结论类目标收束。\n"
                            "5. 必须结合相关记忆理解当前子问题，必要时在目标里保留记忆提供的省份、科类、年份、专业、层次等约束。\n"
                            "6. 不要把工具名、能力名或实现手段写进目标，不要写成“调用 RAG”“调用 MCP”“搜索网页”“抓取页面”这种执行层指令。\n"
                            "7. 如果问题是最新公告、2025/25年录取、分省分专业计划、分数线、位次或需要公开网页核验，可以把“补强或核验公开信息”写成目标，但仍然不要指定工具。\n"
                            "8. 如果问题在问某个人是谁、某个职务由谁担任、某学院/部门负责人是谁、某联系电话或邮箱是什么，而本地知识库文档清单看起来未明显覆盖该事实，应优先把“补充公开证据并核验事实”纳入计划。\n"
                            "9. 如果本地知识库更像招生章程、概览、FAQ 之类通用资料，而问题要求学院页面上的细粒度字段，也应认真考虑补充公开事实核验目标。\n"
                            "10. 计划必须贴合招生咨询场景，保留问题中的关键约束，不要写成泛泛的“查询信息”“处理问题”这种废话。\n"
                            '输出示例：["先收集与该子问题直接相关的可靠证据。","基于已有证据给出当前子问题结论，并明确不确定性。"]'
                        )
                    ),
                ]
            )
            raw_content = str(getattr(response, "content", "") or "").strip()
            parsed = json.loads(raw_content)
            if not isinstance(parsed, list):
                return self._rule_build_plan(query)
            steps: list[PlanStep] = []
            for item in parsed:
                if not isinstance(item, str):
                    continue
                goal = self._normalize_plan_step_goal(item)
                if not goal:
                    continue
                steps.append(PlanStep(goal=goal))
            if not steps:
                return self._rule_build_plan(query)
            if not self._looks_like_synthesis_goal(steps[-1].goal):
                steps.append(PlanStep(goal="基于前面步骤获得的证据，给出当前子问题结论并说明不确定性。"))
            return steps
        except Exception:
            return self._rule_build_plan(query)

    def run_subproblem_agent(
        self,
        *,
        step: PlanStep,
        plan_step_index: int,
        total_plan_steps: int,
        subproblem: SubproblemState,
        request: ChatRequest,
        fail_features: set[str],
        effective_features: list[FeatureFlag],
        memory_context_blocks: list[str],
        memory_text: str,
        trace_id: str,
        route_label: str,
        step_events: list[AgentStepEvent],
        sink: StepSink | None,
        attempt: int,
    ) -> StepExecutionResult:
        llm = build_langchain_chat_model(
            self.deps.services.settings,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        if llm is None:
            raise RuntimeError("agent_llm_unavailable")
        try:
            from langchain_core.messages import HumanMessage
            from langchain_core.tools import tool
            from langgraph.prebuilt import create_react_agent
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("react_agent_dependencies_unavailable") from exc

        collector_context_blocks = list(subproblem.context_blocks)
        collector_sources = list(subproblem.sources)
        collector_notes = list(subproblem.notes)
        collector_tool_audit: list[str] = []
        search_result_url_map: dict[str, str] = {}
        rag_document_catalog = self._get_rag_document_catalog_text()

        def normalize_content(content: Any) -> str:
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                rows: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        rows.append(item.strip())
                    elif isinstance(item, dict) and item.get("type") == "text":
                        rows.append(str(item.get("text", "")).strip())
                return "\n".join(item for item in rows if item).strip()
            return str(content or "").strip()
        runtime = self.get_mcp_runtime(trace_id)
        if runtime.notes:
            collector_notes.extend(runtime.notes)
        tools: list[Any] = []
        tools.extend(
            self._build_local_agent_tools(
                tool_factory=tool,
                request=request,
                effective_features=effective_features,
                fail_features=fail_features,
                memory_context_blocks=memory_context_blocks,
                collector_context_blocks=collector_context_blocks,
                collector_sources=collector_sources,
                collector_notes=collector_notes,
                collector_tool_audit=collector_tool_audit,
                rag_document_catalog=rag_document_catalog,
            )
        )
        runtime_tools, runtime_tool_names = self._build_runtime_agent_tools(
            tool_factory=tool,
            runtime=runtime,
            collector_sources=collector_sources,
            collector_tool_audit=collector_tool_audit,
            search_result_url_map=search_result_url_map,
        )
        tools.extend(runtime_tools)
        if runtime.tools:
            collector_tool_audit.append("agent_tool:mcp_runtime:" + ",".join(item.alias for item in runtime.servers))
        elif runtime.servers:
            collector_tool_audit.append("agent_tool:mcp_runtime_unavailable:" + ",".join(item.alias for item in runtime.servers))

        history_messages = self.gateway._build_langchain_history_messages(request.messages[:-1])
        memory_blocks_text = "\n".join(memory_context_blocks) if memory_context_blocks else "当前没有可用记忆片段。"
        prior_evidence = "\n".join(collector_context_blocks[:8]) if collector_context_blocks else "当前没有已有证据。"
        available_tools = (
                    "\n".join(
                        self._format_agent_tool_prompt_line(item)
                        for item in tools
                    )
                    if tools
                    else "当前步骤不提供额外工具。"
                )
        prompt = (
            "你是中原工学院招生专家模式下负责执行单个计划节点的 ReAct 智能体。"
            "当前工具列表中的所有工具都可以按需使用。"
            "计划步骤只定义本步目标，不限制你必须使用哪一种工具或能力。"
            "如果默认上下文已经足够，可以不调工具直接回答。"
            "你的回答必须聚焦当前计划节点，而不是一次性回答整个子问题。"
            "如果证据不足，明确说不确定，不要编造来源。"
            "你必须把短期记忆、长期记忆、特殊记忆都当作真实可用上下文，并在整个推理过程中持续参考。"
            "不要机械调用工具；只有当已有证据不足、问题要求最新信息、需要公开网页核验、需要外部搜索，或当前问题在询问人物身份、现任职务、院系领导、部门负责人、联系方式等开放网页事实时才调用合适工具。"
            "如果本地资料不足，你应主动判断是否需要外部工具补充证据。"
            "你做事非常负责, 你会尽可能找到信息, 哪怕进行深度思考与多轮无上限次数的工具调用!"
        )
        human_prompt = (
            f"执行策略：{request.agent_strategy}\n"
            f"问题路由：{route_label}\n"
            f"子问题：{subproblem.query}\n\n"
            f"当前计划步骤：第 {plan_step_index} / {total_plan_steps} 步\n"
            f"当前计划步骤目标：{step.goal}\n\n"
            "结构化记忆：\n"
            f"{memory_text or '当前没有可用记忆。'}\n\n"
            "记忆上下文片段：\n"
            f"{memory_blocks_text}\n\n"
            "当前已拿到的历史证据：\n"
            f"{prior_evidence}\n\n"
            "当前可用工具：\n"
            f"{available_tools}\n\n"
            "Tool call example:\n"
            "If search returns uuid/url pairs and you need crawl_webpage, call it like "
            '{"uuids":["uuid-1"],"urlMap":{"uuid-1":"https://example.com/page"}}'
            ".\n\n"
            "5. If a tool exposes args_schema, you must fill every required field before calling it.\n"
            "6. If a crawl tool needs search results, pass both `uuids` and `urlMap`; `urlMap` must be an object mapping each UUID to its URL.\n\n"
            "补充判断规则：\n"
            "1. 如果问题涉及最新公告、近年录取、分省计划、分数线、位次或公开通知，而当前证据不足，可优先考虑 MCP 搜索或抓取工具。\n"
            "2. 如果问题在问某个人是谁、某个职务由谁担任、院系领导或部门负责人是谁、联系电话/邮箱/办公地点是什么，而当前证据不足，也应优先考虑 MCP 搜索或抓取工具。\n"
            "3. 如果本地知识库文档已经足够支撑当前步骤，就不要硬调外部工具。\n"
            "4. 当前步骤的目标优先于工具偏好；工具只是手段，不是目标本身。\n"
            "5. 如果需要调用 local_rag_search，可优先参考工具描述中的文档清单来判断本地库是否可能命中。\n\n"
            "请使用 ReAct 方式执行当前计划节点。"
            "只输出本步骤的执行结果。"
        )

        agent = create_react_agent(llm, tools, prompt=prompt, version="v2")
        result = asyncio.run(
            agent.ainvoke(
                {
                    "messages": [
                        *history_messages,
                        HumanMessage(content=human_prompt),
                    ]
                }
            )
        )
        pending_tool_calls: list[dict[str, Any]] = []
        for message in list(result.get("messages") or []):
            if getattr(message, "type", "") == "ai":
                pending_tool_calls.extend(self._extract_message_tool_calls(message))
                continue
            if getattr(message, "type", "") != "tool":
                continue
            tool_name = str(getattr(message, "name", "") or "")
            tool_call_id = str(getattr(message, "tool_call_id", "") or "")
            content = normalize_content(getattr(message, "content", ""))
            matched_call = self._pop_matching_tool_call(
                pending_tool_calls,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )
            self._append_tool_trace_record(
                trace_id=trace_id,
                session_id=request.session_id,
                tool_name=tool_name,
                tool_args=(matched_call or {}).get("args"),
                tool_output=content or "无返回内容",
                subproblem_id=subproblem.subproblem_id,
                plan_step_index=plan_step_index,
                attempt=attempt,
            )
            self.emit_step(
                events=step_events,
                sink=sink,
                strategy=request.agent_strategy,
                node=f"tool_call_{tool_name or 'unknown_tool'}",
                title=f"工具调用：{tool_name or 'unknown_tool'}",
                status="completed",
                message=self._build_tool_trace_message(
                    tool_name=tool_name,
                    tool_args=(matched_call or {}).get("args"),
                    tool_output=content or "无返回内容",
                ),
                subproblem_id=subproblem.subproblem_id,
                plan_step_index=plan_step_index,
                attempt=attempt,
            )
            if not content:
                continue
            if tool_name in {"local_rag_search", "major_catalog_lookup", "scoreline_lookup", "policy_table_lookup"}:
                continue
            collector_context_blocks.append(f"[tool:{tool_name or 'tool'}] {content}")
            if tool_name in runtime_tool_names:
                collector_tool_audit.append(f"agent_tool:mcp_tool:{tool_name or 'unknown_tool'}")
        output = self._extract_agent_output_text(result)
        if not output:
            raise RuntimeError("subproblem_agent_output_empty")
        collector_context_blocks.append(f"[plan-step:{plan_step_index}] {output}")
        return StepExecutionResult(
            ok=True,
            message=output,
            context_blocks=collector_context_blocks,
            sources=self.dedupe_sources(collector_sources, limit=5),
            tool_audit=collector_tool_audit,
            notes=list(dict.fromkeys(collector_notes)),
        )

    def review_subproblem_result(self, *, route_label: str, result: StepExecutionResult) -> StepReviewResult:
        if not result.ok:
            return StepReviewResult(ok=False, message=result.message or "智能体未返回有效结果。")
        if not (result.message or "").strip():
            return StepReviewResult(ok=False, message="智能体输出为空。")
        needs_evidence = route_label in {"faq", "policy", "process", "time_sensitive"}
        if needs_evidence and not result.context_blocks and not result.sources:
            return StepReviewResult(ok=False, message="智能体未拿到可用证据。")
        return StepReviewResult(ok=True, message="智能体已完成子问题。")

    def _build_local_agent_tools(
        self,
        *,
        tool_factory: Callable[..., Any],
        request: ChatRequest,
        effective_features: list[FeatureFlag],
        fail_features: set[str],
        memory_context_blocks: list[str],
        collector_context_blocks: list[str],
        collector_sources: list[ChatSource],
        collector_notes: list[str],
        collector_tool_audit: list[str],
        rag_document_catalog: str,
    ) -> list[Any]:
        tools: list[Any] = []
        effective_feature_set = set(effective_features)
        structured_toolset = StructuredAdmissionsToolset(self.deps.services.settings)

        if "rag" in effective_feature_set:

            @tool_factory("local_rag_search")
            def local_rag_search(tool_query: str) -> str:
                """检索中原工学院本地知识库，并优先参考当前已收录的招生文档。"""
                collector_tool_audit.append("agent_tool:local_rag_search")
                rag_result = self.deps.container.isolation.execute(
                    "rag-agent-service",
                    lambda: self.gateway._invoke_rag(
                        request.session_id,
                        tool_query,
                        fail_features,
                        memory_context_blocks + collector_context_blocks,
                    ),
                )
                if not rag_result.ok or rag_result.value is None:
                    return f"RAG 检索失败：{rag_result.error or 'unknown'}"
                rag_output = rag_result.value
                collector_context_blocks.extend(rag_output.context_blocks[: self.deps.services.settings.rag_final_top_k])
                collector_sources[:] = self.dedupe_sources(
                    [*collector_sources, *[ChatSource(title=item.title, url=item.url) for item in rag_output.sources]],
                    limit=5,
                )
                if rag_output.degrade_reason:
                    collector_notes.append(f"RAG 降级：{rag_output.degrade_reason}")
                return "\n".join(item.strip() for item in rag_output.context_blocks[:3] if item.strip()) or "未检索到可靠本地资料。"

            local_rag_search.description = (
                "检索中原工学院本地知识库。"
                "当问题与学校概况、学院专业、招生章程、录取规则、学费资助、校园服务等校内资料有关时优先使用。"
                f"当前知识库已收录的文档包括：{rag_document_catalog}"
            )
            tools.append(local_rag_search)

        def run_structured_lookup(tool_name: str, tool_query: str, *, limit: int) -> str:
            collector_tool_audit.append(f"agent_tool:{tool_name}")
            try:
                if tool_name == "major_catalog_lookup":
                    payload = structured_toolset.major_catalog_fulltext()
                elif tool_name == "scoreline_lookup":
                    payload = structured_toolset.scoreline_fulltext()
                elif tool_name == "policy_table_lookup":
                    payload = structured_toolset.policy_table_fulltext()
                else:
                    return f"结构化工具不存在：{tool_name}"
            except Exception as exc:
                return f"结构化检索失败：{exc}"
            if not payload.records:
                return "未检索到匹配的结构化记录。"
            structured_response = structured_toolset.to_rag_response(payload=payload, trace_id=request.session_id)
            if structured_response is not None:
                collector_context_blocks.extend(
                    structured_response.context_blocks[: self.deps.services.settings.rag_final_top_k]
                )
                collector_sources[:] = self.dedupe_sources(
                    [*collector_sources, *[ChatSource(title=item.title, url=item.url) for item in structured_response.sources]],
                    limit=5,
                )
            return structured_toolset.render_payload_text(payload)

        @tool_factory("major_catalog_lookup")
        def major_catalog_lookup(tool_query: str) -> str:
            """查询招生专业目录类结构化知识库。"""
            return run_structured_lookup("major_catalog_lookup", tool_query, limit=8)

        major_catalog_lookup.description = (
            "查询专业目录类结构化知识库（xlsx/xls）。"
            "适用于专业代码、专业名称、学制、学费、选考科目、学位授予门类、所属学院等问题。"
        )
        tools.append(major_catalog_lookup)

        @tool_factory("scoreline_lookup")
        def scoreline_lookup(tool_query: str) -> str:
            """查询录取分数线类结构化知识库。"""
            return run_structured_lookup("scoreline_lookup", tool_query, limit=8)

        scoreline_lookup.description = (
            "查询录取分数线类结构化知识库（xlsx/xls）。"
            "适用于年份、省份、批次、科类、专业最低分、最低位次等问题。"
        )
        tools.append(scoreline_lookup)

        @tool_factory("policy_table_lookup")
        def policy_table_lookup(tool_query: str) -> str:
            """查询招生章程或政策附表类结构化知识库。"""
            return run_structured_lookup("policy_table_lookup", tool_query, limit=12)

        policy_table_lookup.description = (
            "查询政策附表类结构化知识库（xlsx/xls）。"
            "适用于章程附表、专业情况汇总表、政策性结构化表格等问题。"
        )
        tools.append(policy_table_lookup)

        if "skill_exec" in effective_feature_set:

            @tool_factory("general_skill")
            def general_skill(tool_query: str) -> str:
                """调用通用技能处理流程化或结构化子任务。"""
                allowed, reason = self.gateway._guard_skill_request(query=tool_query, saved_skill_id=None)
                collector_tool_audit.append(f"agent_tool:general_skill:{reason}")
                if not allowed:
                    return f"技能执行被拦截：{reason}"
                skill_result = self.deps.container.isolation.execute(
                    "skill-service",
                    lambda: self.gateway._invoke_skill(tool_query, request.session_id, None, fail_features),
                )
                if not skill_result.ok or not skill_result.value:
                    return f"技能执行失败：{skill_result.error or 'unknown'}"
                return str(skill_result.value)

            tools.append(general_skill)

        if "use_saved_skill" in effective_feature_set and request.saved_skill_id:

            @tool_factory("saved_skill")
            def saved_skill(tool_query: str) -> str:
                """调用已保存技能，仅在当前会话存在 saved_skill_id 时可用。"""
                allowed, reason = self.gateway._guard_skill_request(query=tool_query, saved_skill_id=request.saved_skill_id)
                collector_tool_audit.append(f"agent_tool:saved_skill:{reason}")
                if not allowed:
                    return f"历史技能调用被拦截：{reason}"
                skill_result = self.deps.container.isolation.execute(
                    "saved-skill-service",
                    lambda: self.gateway._invoke_skill(tool_query, request.session_id, request.saved_skill_id, fail_features),
                )
                if not skill_result.ok or not skill_result.value:
                    return f"历史技能执行失败：{skill_result.error or 'unknown'}"
                return str(skill_result.value)

            tools.append(saved_skill)

        return tools

    def _build_runtime_agent_tools(
        self,
        *,
        tool_factory: Callable[..., Any],
        runtime: McpToolRuntime,
        collector_sources: list[ChatSource],
        collector_tool_audit: list[str],
        search_result_url_map: dict[str, str],
    ) -> tuple[list[Any], set[str]]:
        if not runtime.tools:
            return [], set()

        normalized_runtime_tools = [self._normalize_agent_tool(item) for item in runtime.tools]
        runtime_tools: list[Any] = []

        for runtime_tool in normalized_runtime_tools:
            tool_name = self._get_agent_tool_name(runtime_tool)
            if tool_name == "bing_search":
                runtime_tools.append(
                    self._build_wrapped_bing_search_tool(
                        tool_factory=tool_factory,
                        runtime_tool=runtime_tool,
                        collector_sources=collector_sources,
                        search_result_url_map=search_result_url_map,
                    )
                )
                continue
            if tool_name == "crawl_webpage":
                runtime_tools.append(
                    self._build_wrapped_crawl_webpage_tool(
                        tool_factory=tool_factory,
                        runtime_tool=runtime_tool,
                        collector_sources=collector_sources,
                        search_result_url_map=search_result_url_map,
                    )
                )
                continue
            runtime_tools.append(runtime_tool)

        runtime_tool_names = {self._get_agent_tool_name(item) for item in normalized_runtime_tools}
        return runtime_tools, runtime_tool_names

    def _build_wrapped_bing_search_tool(
        self,
        *,
        tool_factory: Callable[..., Any],
        runtime_tool: Any,
        collector_sources: list[ChatSource],
        search_result_url_map: dict[str, str],
    ) -> Any:
        @tool_factory("bing_search")
        async def bing_search(query: str, count: int = 10, offset: int = 0) -> str:
            """调用 Bing 搜索工具检索公开网页结果。"""
            result = await self._invoke_mcp_tool(
                runtime_tool,
                {
                    "query": query,
                    "count": count,
                    "offset": offset,
                },
            )
            search_result_url_map.update(self._extract_uuid_url_map(result))
            collector_sources[:] = self.dedupe_sources(
                [*collector_sources, *self._extract_chat_sources_from_tool_result(result)],
                limit=5,
            )
            return self._normalize_tool_result_text(result) or "未获取到搜索结果。"

        bing_search.description = self._get_agent_tool_description(runtime_tool)
        return bing_search

    def _build_wrapped_crawl_webpage_tool(
        self,
        *,
        tool_factory: Callable[..., Any],
        runtime_tool: Any,
        collector_sources: list[ChatSource],
        search_result_url_map: dict[str, str],
    ) -> Any:
        @tool_factory("crawl_webpage")
        async def crawl_webpage(uuids: list[str], urlMap: dict[str, str] | None = None) -> str:
            """根据搜索结果里的 UUID 和 URL 抓取网页正文。"""
            payload = self._build_crawl_webpage_payload(
                uuids=uuids,
                url_map=urlMap,
                cached_url_map=search_result_url_map,
            )
            result = await self._invoke_mcp_tool(runtime_tool, payload)
            collector_sources[:] = self.dedupe_sources(
                [*collector_sources, *self._extract_chat_sources_from_tool_result(result)],
                limit=5,
            )
            return self._normalize_tool_result_text(result) or "未获取到网页抓取结果。"

        crawl_webpage.description = self._get_agent_tool_description(runtime_tool)
        return crawl_webpage

    async def _invoke_mcp_tool(self, tool: Any, query: str) -> Any:
        if hasattr(tool, "ainvoke"):
            try:
                return await tool.ainvoke(query)
            except Exception:
                return await tool.ainvoke({"query": query})
        if hasattr(tool, "invoke"):
            try:
                return tool.invoke(query)
            except Exception:
                return tool.invoke({"query": query})
        raise RuntimeError("mcp_tool_not_invokable")

    def review_step(
        self,
        step: PlanStep,
        result: StepExecutionResult,
        *,
        is_final_step: bool,
        accumulated_tool_audit: list[str] | None = None,
    ) -> StepReviewResult:
        if not result.ok:
            return StepReviewResult(ok=False, message=result.message or "步骤执行未返回有效结果。")
        if not (result.message or "").strip():
            return StepReviewResult(ok=False, message="步骤执行结果为空。")
        if is_final_step:
            normalized = (result.message or "").strip()
            unresolved_markers = (
                "不能据此确定",
                "无法确认",
                "目前无法确认",
                "暂未获得可靠依据",
                "不能确定",
                "暂时无法确认",
                "证据不足",
            )
            used_mcp = any(
                item.startswith("agent_tool:mcp_tool:")
                for item in [*(accumulated_tool_audit or []), *result.tool_audit]
            )
            if self._has_mcp_servers() and not used_mcp and any(marker in normalized for marker in unresolved_markers):
                return StepReviewResult(ok=False, message="当前结论仍未确认，且尚未使用 MCP 外部工具补强。")
        return StepReviewResult(ok=True, message="步骤满足要求。")

    def replan_subproblem(self, subproblem: SubproblemState, request: ChatRequest, memory_text: str = "") -> SubproblemState:
        route_label, _ = self.gateway._classify_query_intent(subproblem.query)
        replanned = SubproblemState(
            subproblem_id=subproblem.subproblem_id,
            query=subproblem.query,
            plan_steps=self.build_plan(
                query=subproblem.query,
                effective_features=request.features,
                route_label=route_label,
                request=request,
                strategy="quality",
                memory_text=memory_text,
            ),
            replan_count=subproblem.replan_count + 1,
        )
        return replanned

    def build_final_session(
        self,
        *,
        request: ChatRequest,
        trace_id: str,
        last_user: str,
        final_text: str,
        sources: list[ChatSource],
        tool_audit: list[str],
        degraded_features: list[FeatureFlag],
        step_events: list[AgentStepEvent],
        status_override: str | None = None,
        error_message: str | None = None,
    ) -> SessionResult:
        prefix_text, degraded = self.gateway._build_citation_notice(request.features, sources, degraded_features)
        merged_text = f"{prefix_text}{final_text}"
        output_flagged, output_reason, audited_text = self.audit_generated_output(merged_text)
        status = status_override or "ok"
        if output_flagged:
            tool_audit = [*tool_audit, f"safety_audit:output_sanitized:{output_reason}"]
            merged_text = audited_text
            status = "failed"
        self.gateway._persist_memory_side_effects(request.session_id, last_user, merged_text)
        return SessionResult(
            session_id=request.session_id,
            trace_id=trace_id,
            text=merged_text,
            status=status,  # type: ignore[arg-type]
            degraded_features=[],
            sources=self.dedupe_sources(sources, limit=5),
            tool_audit=list(dict.fromkeys(tool_audit)),
            error_message=error_message,
            agent_strategy=request.agent_strategy,
            agent_trace=list(step_events),
        )

    def generate_answer(
        self,
        *,
        request: ChatRequest,
        query: str,
        context_blocks: list[str],
        feature_notes: list[str],
        fail_features: set[str],
    ) -> str:
        generation_output = self.gateway._invoke_generation(
            user_query=query,
            context_blocks=context_blocks,
            feature_notes=feature_notes,
            request=request,
            fail_features=fail_features,
        )
        return generation_output.text

    def should_degrade_generation_error(self, error_message: str | None) -> bool:
        normalized = (error_message or "").strip().lower()
        if not normalized:
            return False
        soft_tokens = (
            "timed out",
            "timeout",
            "circuit_open:generation-service",
        )
        hard_tokens = (
            "generation failure injected",
        )
        if any(token in normalized for token in hard_tokens):
            return False
        return any(token in normalized for token in soft_tokens)

    def compact_generation_context(self, *, context_blocks: list[str], strategy: AgentStrategy) -> list[str]:
        limit = 4 if strategy == "speed" else 6
        compacted: list[str] = []
        seen: set[str] = set()
        for item in context_blocks:
            normalized = self._compact_text_block(item, limit_chars=220)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            compacted.append(normalized)
            if len(compacted) >= limit:
                break
        return compacted

    def compact_feature_notes(self, *, notes: list[str], strategy: AgentStrategy) -> list[str]:
        limit = 6 if strategy == "speed" else 8
        compacted: list[str] = []
        seen: set[str] = set()
        for item in notes:
            normalized = self._compact_text_block(item, limit_chars=120)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            compacted.append(normalized)
            if len(compacted) >= limit:
                break
        return compacted

    def build_rule_based_final_answer(
        self,
        *,
        query: str,
        context_blocks: list[str],
        notes: list[str],
        error_message: str,
    ) -> str:
        rows = [
            "当前最终生成阶段超时，以下为基于已完成步骤的保守汇总：",
            f"问题：{query}",
            "",
        ]
        evidence_rows = [item.strip() for item in context_blocks if item.strip()][:5]
        note_rows = [item.strip() for item in notes if item.strip()][:5]
        if evidence_rows:
            rows.append("已取得的依据：")
            rows.extend(f"- {item[:180]}" for item in evidence_rows)
            rows.append("")
        if note_rows:
            rows.append("执行备注：")
            rows.extend(f"- {item[:160]}" for item in note_rows)
            rows.append("")
        rows.append(f"失败原因：{error_message}")
        rows.append("建议：稍后重试，或切换更快模型/速度优先策略后再试。")
        return "\n".join(rows).strip()

    def _rule_rewrite_query(
        self,
        *,
        last_user: str,
        memory_text: str,
        strategy: AgentStrategy,
    ) -> str:
        normalized = " ".join(last_user.split()).strip()
        if strategy == "quality" and memory_text and any(token in normalized for token in ("这个", "那个", "它", "那")):
            first_memory_line = next((line[2:] for line in memory_text.splitlines() if line.startswith("- ")), "")
            if first_memory_line:
                return f"{normalized}（结合上下文：{first_memory_line}）"
        return normalized

    def _build_recent_user_context(self, *, request: ChatRequest, max_messages: int = 3) -> str:
        user_messages = [
            re.sub(r"\s+", " ", str(item.content or "")).strip()
            for item in request.messages
            if getattr(item, "role", "") == "user" and str(item.content or "").strip()
        ]
        if not user_messages:
            return "当前没有可用的最近用户上下文。"
        recent = user_messages[-max_messages:]
        return "\n".join(f"- {item}" for item in recent)

    def _compact_text_block(self, text: str, *, limit_chars: int) -> str:
        normalized = re.sub(r"\s+", " ", (text or "")).strip()
        if len(normalized) <= limit_chars:
            return normalized
        keep = normalized[: limit_chars - 3].rstrip(" ，。；;,:：")
        return f"{keep}..."

    def _rule_split_query(self, query: str, *, max_subproblems: int) -> list[str]:
        parts = [item.strip(" ，。；;？?！!") for item in re.split(r"[；;。]|(?:并且)|(?:以及)|(?:同时)|(?:另外)", query)]
        parts = [item for item in parts if item]
        if not parts:
            return [query]
        return parts[:max_subproblems]

    def _normalize_plan_step_goal(self, value: str) -> str:
        normalized = re.sub(r"\s+", " ", str(value or "")).strip().strip(" -\t\r\n")
        return normalized

    def _looks_like_synthesis_goal(self, goal: str) -> bool:
        normalized = self._normalize_plan_step_goal(goal)
        if not normalized:
            return False
        markers = (
            "综合",
            "总结",
            "结论",
            "回答",
            "判断",
            "说明不确定性",
            "给出当前子问题答案",
        )
        return any(token in normalized for token in markers)

    def _rule_build_plan(self, query: str) -> list[PlanStep]:
        normalized_query = self._normalize_plan_step_goal(query) or "当前子问题"
        return [
            PlanStep(goal=f"先收集与“{normalized_query}”直接相关的可靠证据，确认回答所需的关键事实与约束。"),
            PlanStep(goal=f"基于前面步骤获得的证据回答“{normalized_query}”，并明确结论边界与不确定性。"),
        ]

    def _format_plan_step_title(self, step: PlanStep, step_index: int) -> str:
        goal = self._normalize_plan_step_goal(step.goal)
        compact = self._compact_text_block(goal, limit_chars=22)
        return compact or f"计划步骤 {step_index}"

    def _extract_agent_output_text(self, result: Any) -> str:
        messages = list(result.get("messages") or []) if isinstance(result, dict) else []
        for message in reversed(messages):
            if getattr(message, "type", "") != "ai":
                continue
            content = getattr(message, "content", "")
            if isinstance(content, str):
                normalized = content.strip()
                if normalized:
                    return normalized
                continue
            if isinstance(content, list):
                rows: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        rows.append(item.strip())
                        continue
                    if isinstance(item, dict) and item.get("type") == "text":
                        rows.append(str(item.get("text", "")).strip())
                normalized = "\n".join(item for item in rows if item)
                if normalized.strip():
                    return normalized.strip()
        return ""

    def _normalize_agent_tool(self, tool: Any) -> Any:
        bound_tool = getattr(tool, "bound", None)
        if getattr(tool, "name", None) is None and getattr(bound_tool, "name", None):
            return bound_tool
        return tool

    def _get_agent_tool_name(self, tool: Any) -> str:
        normalized_tool = self._normalize_agent_tool(tool)
        name = str(getattr(normalized_tool, "name", "") or "").strip()
        if name:
            return name
        return str(getattr(tool, "name", "") or "").strip() or "unknown_tool"

    def _get_agent_tool_description(self, tool: Any) -> str:
        normalized_tool = self._normalize_agent_tool(tool)
        description = str(getattr(normalized_tool, "description", "") or "").strip()
        if description:
            return description
        return str(getattr(tool, "description", "") or "").strip()

    def _get_agent_tool_args_schema_text(self, tool: Any) -> str:
        normalized_tool = self._normalize_agent_tool(tool)
        args_schema = getattr(normalized_tool, "args_schema", None)
        if args_schema is None:
            return ""
        if isinstance(args_schema, (dict, list)):
            rendered = json.dumps(args_schema, ensure_ascii=False)
        else:
            rendered = str(args_schema)
        return self._compact_text_block(rendered, limit_chars=320)

    def _format_agent_tool_prompt_line(self, tool: Any) -> str:
        name = self._get_agent_tool_name(tool)
        description = self._get_agent_tool_description(tool) or "无额外描述。"
        args_schema_text = self._get_agent_tool_args_schema_text(tool)
        if args_schema_text:
            return f"- {name}: {description} | args_schema={args_schema_text}"
        return f"- {name}: {description}"

    def _normalize_tool_result_text(self, result: Any) -> str:
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, list):
            rows: list[str] = []
            for item in result:
                if isinstance(item, str):
                    rows.append(item.strip())
                    continue
                if isinstance(item, dict) and item.get("type") == "text":
                    rows.append(str(item.get("text", "")).strip())
                    continue
                if isinstance(item, dict):
                    rows.append(json.dumps(item, ensure_ascii=False))
            return "\n".join(item for item in rows if item).strip()
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
        return str(result or "").strip()

    def _extract_json_payloads_from_tool_result(self, result: Any) -> list[Any]:
        payloads: list[Any] = []

        def visit(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, str):
                normalized = value.strip()
                if not normalized or normalized[:1] not in "[{":
                    return
                try:
                    payloads.append(json.loads(normalized))
                except Exception:
                    return
                return
            if isinstance(value, list):
                for item in value:
                    visit(item)
                return
            if isinstance(value, dict):
                if value.get("type") == "text":
                    visit(value.get("text"))
                    return
                payloads.append(value)

        visit(result)
        return payloads

    def _extract_uuid_url_map(self, result: Any) -> dict[str, str]:
        extracted: dict[str, str] = {}

        def walk(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    walk(item)
                return
            if isinstance(value, dict):
                uuid_value = str(value.get("uuid", "") or "").strip()
                url_value = str(value.get("url", "") or "").strip()
                if uuid_value and url_value:
                    extracted[uuid_value] = url_value
                for nested in value.values():
                    walk(nested)

        for payload in self._extract_json_payloads_from_tool_result(result):
            walk(payload)
        return extracted

    def _extract_chat_sources_from_tool_result(self, result: Any) -> list[ChatSource]:
        sources: list[ChatSource] = []
        seen: set[tuple[str, str]] = set()

        def walk(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    walk(item)
                return
            if isinstance(value, dict):
                url_value = str(value.get("url", "") or "").strip()
                title_value = str(value.get("title", "") or value.get("name", "") or url_value).strip()
                if url_value:
                    key = (title_value, url_value)
                    if key not in seen:
                        seen.add(key)
                        sources.append(ChatSource(title=title_value or url_value, url=url_value))
                for nested in value.values():
                    walk(nested)

        for payload in self._extract_json_payloads_from_tool_result(result):
            walk(payload)
        return sources

    def _build_crawl_webpage_payload(
        self,
        *,
        uuids: list[str],
        url_map: dict[str, str] | None,
        cached_url_map: dict[str, str],
    ) -> dict[str, Any]:
        normalized_uuids = [str(item).strip() for item in uuids if str(item).strip()]
        if not normalized_uuids:
            raise RuntimeError("crawl_webpage_requires_uuids")
        merged_url_map = {
            str(key).strip(): str(value).strip()
            for key, value in dict(url_map or {}).items()
            if str(key).strip() and str(value).strip()
        }
        for uuid_value in normalized_uuids:
            if uuid_value not in merged_url_map and cached_url_map.get(uuid_value):
                merged_url_map[uuid_value] = cached_url_map[uuid_value]
        missing = [uuid_value for uuid_value in normalized_uuids if uuid_value not in merged_url_map]
        if missing:
            raise RuntimeError(f"crawl_webpage_missing_urlmap:{','.join(missing)}")
        return {
            "uuids": normalized_uuids,
            "urlMap": {uuid_value: merged_url_map[uuid_value] for uuid_value in normalized_uuids},
        }

    def get_mcp_runtime(self, trace_id: str) -> McpToolRuntime:
        with self._mcp_runtime_guard:
            cached = self._mcp_runtime_cache.get(trace_id)
            if cached is not None:
                self.logger.info("[agent.mcp] trace=%s cache_hit tools=%s", trace_id, len(cached.tools))
                return cached

        started_at = perf_counter()
        runtime = asyncio.run(build_langchain_mcp_runtime(self.deps.services.settings))
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        self.logger.info(
            "[agent.mcp] trace=%s runtime_ready elapsed_ms=%s tools=%s servers=%s",
            trace_id,
            elapsed_ms,
            len(runtime.tools),
            len(runtime.servers),
        )
        with self._mcp_runtime_guard:
            cached = self._mcp_runtime_cache.get(trace_id)
            if cached is not None:
                started_at = perf_counter()
                asyncio.run(runtime.aclose())
                close_elapsed_ms = int((perf_counter() - started_at) * 1000)
                self.logger.info(
                    "[agent.mcp] trace=%s duplicate_runtime_closed elapsed_ms=%s",
                    trace_id,
                    close_elapsed_ms,
                )
                return cached
            self._mcp_runtime_cache[trace_id] = runtime
            self._mcp_runtime_locks.setdefault(trace_id, threading.RLock())
            return runtime

    def release_mcp_runtime(self, trace_id: str) -> None:
        runtime: McpToolRuntime | None = None
        with self._mcp_runtime_guard:
            runtime = self._mcp_runtime_cache.pop(trace_id, None)
            self._mcp_runtime_locks.pop(trace_id, None)
        if runtime is None:
            return
        started_at = perf_counter()
        asyncio.run(runtime.aclose())
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        self.logger.info("[agent.mcp] trace=%s runtime_released elapsed_ms=%s", trace_id, elapsed_ms)

    def _get_mcp_runtime_lock(self, trace_id: str) -> threading.RLock:
        with self._mcp_runtime_guard:
            lock = self._mcp_runtime_locks.get(trace_id)
            if lock is None:
                lock = threading.RLock()
                self._mcp_runtime_locks[trace_id] = lock
            return lock
