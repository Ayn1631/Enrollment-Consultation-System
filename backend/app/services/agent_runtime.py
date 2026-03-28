from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import uuid
from datetime import datetime
from time import perf_counter
from typing import Any, Callable

from app.models import AgentStepEvent, AgentStrategy, ChatRequest, ChatSource, FeatureFlag, SessionResult
from app.services.ai_stack import (
    McpToolRuntime,
    build_langchain_chat_model,
    build_langchain_mcp_runtime,
    load_mcp_server_configs,
)
from app.services.agent_types import (
    PlanStep,
    PlanStepType,
    StepExecutionResult,
    StepReviewResult,
    SubproblemState,
)


StepSink = Callable[[AgentStepEvent], None]


class AgentRuntime:
    def __init__(self, gateway: Any):
        self.gateway = gateway
        self.logger = logging.getLogger(__name__)
        self._mcp_runtime_cache: dict[str, McpToolRuntime] = {}
        self._mcp_runtime_locks: dict[str, threading.RLock] = {}
        self._mcp_runtime_guard = threading.RLock()

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

    def rewrite_query(
        self,
        *,
        request: ChatRequest,
        last_user: str,
        memory_text: str,
        strategy: AgentStrategy,
    ) -> str:
        llm = build_langchain_chat_model(
            self.deps.services.settings,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        if llm is None:
            return self._rule_rewrite_query(last_user=last_user, memory_text=memory_text, strategy=strategy)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except Exception:
            return self._rule_rewrite_query(last_user=last_user, memory_text=memory_text, strategy=strategy)
        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "你是“招生咨询查询改写器”。"
                            "你的任务是把用户当前这句话，改写成适合招生咨询工具链执行的单轮查询。"
                            "你的输出会直接用于后续检索、RAG、技能执行、官方搜索或工具路由，因此改写结果必须清晰、完整、可检索、少歧义。"
                            "你必须先在内部综合理解“执行策略、用户原问题、相关记忆”，再给出最终改写结果，但不要输出你的分析过程。"
                            "你只能输出最终改写后的问题文本，不要输出解释、前缀、编号、引号、Markdown、多个候选版本或任何额外内容。"
                            "改写目标如下："
                            "第一，补全省略代词、指代对象、年份、省份、专业、费用项、流程对象等关键缺失信息，使问题变成可以独立理解的单轮查询。"
                            "第二，尽量利用相关记忆补全上下文，但只能使用与当前问题直接相关、且可以高置信度承接的信息。"
                            "第三，必须保持用户原意，不得编造新需求，不得扩写成用户没有表达过的目标。"
                            "第四，如果记忆里存在多个可能指代对象且无法确定，就保持原问题核心意图并做最小保守补全，不要瞎猜。"
                            "第五，不要回答问题本身，你的职责只是改写查询，不是提供结论。"
                            "第六，输出必须是适合工具链理解的一条自然语言查询，而不是搜索语法、标签列表或问答摘要。"
                            "第七，当用户问题已经足够清晰时，只做轻量规范化，不要过度改写。"
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
                            f"相关记忆：\n{memory_text}\n\n"
                            "输出要求：\n"
                            "1. 只输出一条改写后的查询文本。\n"
                            "2. 优先补全当前问题中的省略指代、时间范围、地区、专业、费用项、流程对象等关键缺失信息。\n"
                            "3. 可以根据相关记忆替换“这个、那个、它、那、今年、该专业、这个费用”等模糊表达，但前提是记忆与当前问题直接相关且指向明确。\n"
                            "4. 不要编造用户没有提出的新目标，不要把一个问题扩成多个问题。\n"
                            "5. 不要输出解释，不要输出分析过程，不要输出多个候选版本。\n"
                            "6. 如果原问题已经足够清晰，就做轻量规范化后直接输出。\n"
                            "7. 如果记忆不足以安全替换代词，就保守改写，不要乱猜。"
                        )
                    ),
                ]
            )
            content = str(getattr(response, "content", "") or "").strip()
            return content or self._rule_rewrite_query(last_user=last_user, memory_text=memory_text, strategy=strategy)
        except Exception:
            return self._rule_rewrite_query(last_user=last_user, memory_text=memory_text, strategy=strategy)

    def split_query(
        self,
        query: str,
        strategy: AgentStrategy,
        request: ChatRequest | None = None,
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

请根据上面的约束拆分子问题。
要求：
1. 只返回 JSON 数组。
2. 每个元素都是可独立求解的字符串子问题。
3. 如有必要，优先输出前置问题，再输出深化问题。
4. 不要编造新问题，不要遗漏关键约束。
5. 如果原问题不适合拆分，就返回单元素数组。

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
    ) -> list[PlanStep]:
        allowed_step_types: list[PlanStepType] = []
        if "rag" in effective_features:
            allowed_step_types.append("local_rag_search")
        if self._should_use_mcp(query=query, route_label=route_label, strategy=strategy):
            allowed_step_types.append("mcp_execute")
        allowed_step_types.append("synthesize_step")

        llm = build_langchain_chat_model(
            self.deps.services.settings,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        if llm is None:
            return self._rule_build_plan(allowed_step_types)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except Exception:
            return self._rule_build_plan(allowed_step_types)
        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "你是一个子问题执行计划制定器。"
                            "你需要先理解当前子问题，再输出一个按顺序执行的计划。"
                            "同一子问题内计划节点顺序执行，不同子问题之间会并发执行。"
                            "计划中的每个节点后续都会交给 ReAct Agent 单独执行。"
                            "因此你要优先安排证据收集，再安排结论综合。"
                            "你只能使用允许的 step_type，不得发明新类型。"
                            "最后一步必须是 synthesize_step。"
                            "只输出 JSON 数组，每个元素必须是对象，包含 step_type、title、instruction 三个字段。"
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"执行策略：{strategy}\n"
                            f"问题路由：{route_label}\n"
                            f"子问题：{query}\n"
                            f"允许的 step_type：{json.dumps(allowed_step_types, ensure_ascii=False)}\n\n"
                            "请给出顺序计划。\n"
                            "要求：\n"
                            "1. 只返回 JSON 数组。\n"
                            "2. 步骤数量尽量精简但有效。\n"
                            "3. title 要短，instruction 要明确说明该步骤要完成什么。\n"
                            "4. 如果需要先检索校内资料，再调用 MCP 外部工具，再汇总结论，就按这个顺序输出。\n"
                            '示例：[{"step_type":"local_rag_search","title":"检索校内资料","instruction":"检索与该子问题直接相关的校内资料，提取可用证据。"},{"step_type":"synthesize_step","title":"综合结论","instruction":"基于已有证据给出当前子问题结论，并明确不确定性。"}]'
                        )
                    ),
                ]
            )
            raw_content = str(getattr(response, "content", "") or "").strip()
            parsed = json.loads(raw_content)
            if not isinstance(parsed, list):
                return self._rule_build_plan(allowed_step_types)
            steps: list[PlanStep] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                step_type = str(item.get("step_type", "")).strip()
                if step_type not in allowed_step_types:
                    continue
                title = str(item.get("title", "")).strip() or self._default_plan_step_title(step_type)
                instruction = str(item.get("instruction", "")).strip() or self._default_plan_step_instruction(step_type)
                steps.append(PlanStep(step_type=step_type, title=title, instruction=instruction))
            if not steps:
                return self._rule_build_plan(allowed_step_types)
            if steps[-1].step_type != "synthesize_step":
                steps.append(
                    PlanStep(
                        step_type="synthesize_step",
                        title=self._default_plan_step_title("synthesize_step"),
                        instruction=self._default_plan_step_instruction("synthesize_step"),
                    )
                )
            return steps
        except Exception:
            return self._rule_build_plan(allowed_step_types)

    def run_subproblem_agent(
        self,
        *,
        step: PlanStep,
        subproblem: SubproblemState,
        request: ChatRequest,
        fail_features: set[str],
        effective_features: list[FeatureFlag],
        memory_context_blocks: list[str],
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
        collector_web_hits = list(subproblem.web_hits)

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

        @tool("local_rag_search")
        def local_rag_search(tool_query: str) -> str:
            """检索中原工学院本地知识库。"""
            collector_tool_audit.append("agent_tool:local_rag_search")
            if "rag" not in effective_features:
                return "当前会话未开启 rag 功能。"
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

        runtime = self.get_mcp_runtime(trace_id)
        if runtime.notes:
            collector_notes.extend(runtime.notes)
        tools: list[Any] = []
        if step.step_type == "local_rag_search" and "rag" in effective_features:
            tools.append(local_rag_search)
        if step.step_type == "mcp_execute" and runtime.tools:
            tools.extend(runtime.tools)
            collector_tool_audit.append("agent_tool:mcp_runtime:" + ",".join(item.alias for item in runtime.servers))
        elif step.step_type == "mcp_execute" and runtime.servers:
            collector_tool_audit.append("agent_tool:mcp_runtime_unavailable:" + ",".join(item.alias for item in runtime.servers))

        history_messages = self.gateway._build_langchain_history_messages(request.messages[:-1])
        memory_text = "\n".join(memory_context_blocks[:8]) if memory_context_blocks else "当前没有可用记忆。"
        prior_evidence = "\n".join(collector_context_blocks[:8]) if collector_context_blocks else "当前没有已有证据。"
        available_tools = "\n".join(f"- {getattr(item, 'name', step.step_type)}" for item in tools) if tools else "当前步骤不提供额外工具。"
        prompt = (
            "你是中原工学院招生专家模式下负责执行单个计划节点的 ReAct 智能体。"
            "你只能使用当前计划节点允许的工具。"
            "如果默认上下文已经足够，可以不调工具直接回答。"
            "你的回答必须聚焦当前计划节点，而不是一次性回答整个子问题。"
            "如果证据不足，明确说不确定，不要编造来源。"
        )
        human_prompt = (
            f"执行策略：{request.agent_strategy}\n"
            f"问题路由：{route_label}\n"
            f"子问题：{subproblem.query}\n\n"
            f"当前计划节点：{step.title}\n"
            f"当前计划节点目标：{step.instruction or step.title}\n\n"
            "默认记忆上下文：\n"
            f"{memory_text}\n\n"
            "当前已拿到的历史证据：\n"
            f"{prior_evidence}\n\n"
            "当前可用工具：\n"
            f"{available_tools}\n\n"
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
        for message in list(result.get("messages") or []):
            if getattr(message, "type", "") != "tool":
                continue
            tool_name = str(getattr(message, "name", "") or "")
            content = normalize_content(getattr(message, "content", ""))
            if not content:
                continue
            if tool_name == "local_rag_search":
                continue
            collector_context_blocks.append(f"[mcp:{tool_name or 'tool'}] {content}")
            collector_tool_audit.append(f"agent_tool:mcp_execute:{tool_name or 'unknown_tool'}")
        output = self._extract_agent_output_text(result)
        if not output:
            raise RuntimeError("subproblem_agent_output_empty")
        collector_context_blocks.append(f"[plan-step:{step.step_type}] {output}")
        return StepExecutionResult(
            ok=True,
            message=output,
            context_blocks=collector_context_blocks,
            sources=self.dedupe_sources(collector_sources, limit=5),
            tool_audit=collector_tool_audit,
            notes=list(dict.fromkeys(collector_notes)),
            web_hits=collector_web_hits,
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

    def _rule_build_plan(self, allowed_step_types: list[PlanStepType]) -> list[PlanStep]:
        steps: list[PlanStep] = []
        for step_type in allowed_step_types:
            if step_type == "synthesize_step":
                continue
            steps.append(
                PlanStep(
                    step_type=step_type,
                    title=self._default_plan_step_title(step_type),
                    instruction=self._default_plan_step_instruction(step_type),
                )
            )
        steps.append(
            PlanStep(
                step_type="synthesize_step",
                title=self._default_plan_step_title("synthesize_step"),
                instruction=self._default_plan_step_instruction("synthesize_step"),
            )
        )
        return steps

    def _default_plan_step_title(self, step_type: PlanStepType) -> str:
        title_map: dict[PlanStepType, str] = {
            "recall_memory": "读取会话记忆",
            "local_rag_search": "检索校内资料",
            "official_web_search": "官方联网搜索",
            "official_web_read": "阅读官方网页",
            "general_skill": "执行通用技能",
            "saved_skill": "执行历史技能",
            "mcp_discover": "查看 MCP 工具目录",
            "mcp_execute": "调用 MCP 外部工具",
            "citation_guard": "引用校验",
            "synthesize_step": "综合当前结论",
        }
        return title_map.get(step_type, step_type)

    def _default_plan_step_instruction(self, step_type: PlanStepType) -> str:
        instruction_map: dict[PlanStepType, str] = {
            "recall_memory": "基于默认记忆上下文澄清与当前子问题相关的历史信息。",
            "local_rag_search": "检索与当前子问题直接相关的校内资料，并提取可用证据。",
            "official_web_search": "搜索与当前子问题相关的官方公开信息。",
            "official_web_read": "阅读已命中的官方网页并提取关键信息。",
            "general_skill": "调用通用技能处理当前子问题中的流程或结构化任务。",
            "saved_skill": "调用已保存技能处理当前子问题。",
            "mcp_discover": "查看可用的 MCP 工具目录，确认外部工具能力。",
            "mcp_execute": "调用合适的 MCP 外部工具补充当前子问题所需证据。",
            "citation_guard": "检查当前证据是否足以支撑回答。",
            "synthesize_step": "基于前面步骤获得的证据，给出当前子问题结论并说明不确定性。",
        }
        return instruction_map.get(step_type, step_type)

    def execute_plan_step(
        self,
        *,
        step: PlanStep,
        subproblem: SubproblemState,
        request: ChatRequest,
        fail_features: set[str],
        effective_features: list[FeatureFlag],
        memory_context_blocks: list[str],
        trace_id: str,
    ) -> StepExecutionResult:
        step_type = step.step_type
        if step_type == "recall_memory":
            _, memory_text, notes = self.load_memory_context(request.session_id)
            return StepExecutionResult(ok=bool(memory_text.strip()), message=memory_text, notes=notes)
        if step_type == "local_rag_search":
            if "rag" not in effective_features:
                return StepExecutionResult(ok=False, message="当前会话未开启 rag 功能。")
            rag_result = self.deps.container.isolation.execute(
                "rag-agent-service",
                lambda: self.gateway._invoke_rag(
                    request.session_id,
                    subproblem.query,
                    fail_features,
                    memory_context_blocks + subproblem.context_blocks,
                ),
            )
            if not rag_result.ok or rag_result.value is None:
                return StepExecutionResult(ok=False, message=f"RAG 检索失败：{rag_result.error or 'unknown'}")
            rag_output = rag_result.value
            notes = []
            if rag_output.degrade_reason:
                notes.append(f"RAG 降级：{rag_output.degrade_reason}")
            return StepExecutionResult(
                ok=bool(rag_output.context_blocks),
                message="\n".join(rag_output.context_blocks[:3]) or "未检索到可靠本地资料。",
                context_blocks=rag_output.context_blocks[: self.deps.services.settings.rag_final_top_k],
                sources=self.dedupe_sources(
                    [ChatSource(title=item.title, url=item.url) for item in rag_output.sources],
                    limit=5,
                ),
                notes=notes,
            )
        if step_type == "official_web_search":
            if "web_search" not in effective_features:
                return StepExecutionResult(ok=False, message="当前会话未开启 web_search 功能。")
            allowed, guarded_query, reason = self.gateway._guard_web_search(subproblem.query)
            if not allowed:
                return StepExecutionResult(ok=False, message=f"联网搜索被拦截：{reason}", tool_audit=[f"web_search:blocked:{reason}"])
            search_result = self.deps.container.isolation.execute(
                "web-search-service",
                lambda: self.gateway._invoke_web_search(guarded_query, fail_features),
            )
            if not search_result.ok or not search_result.value:
                return StepExecutionResult(ok=False, message=f"联网搜索失败：{search_result.error or 'unknown'}")
            hits = search_result.value
            return StepExecutionResult(
                ok=True,
                message="\n".join(f"{item.title}: {item.snippet}" for item in hits[:2]),
                context_blocks=[f"联网搜索摘要：{item.title} | {item.snippet}" for item in hits],
                sources=self.dedupe_sources([ChatSource(title=item.title, url=item.url) for item in hits], limit=5),
                tool_audit=["web_search:allowed:official_whitelist"],
                web_hits=[{"title": item.title, "url": item.url, "snippet": item.snippet} for item in hits],
            )
        if step_type == "official_web_read":
            if not subproblem.web_hits:
                return StepExecutionResult(ok=False, message="当前没有可读的官方网页结果。")
            hits = [self.gateway.WebSearchHit(**item) for item in subproblem.web_hits] if hasattr(self.gateway, "WebSearchHit") else []
            if not hits:
                hits = [type("Hit", (), item) for item in subproblem.web_hits]
            read_result = self.deps.container.isolation.execute(
                "web-read-service",
                lambda: self.gateway._invoke_web_read(query=subproblem.query, hits=hits, fail_features=fail_features),
            )
            if not read_result.ok or not read_result.value:
                return StepExecutionResult(
                    ok=False,
                    message=f"网页阅读失败：{read_result.error or 'unknown'}",
                    tool_audit=["web_read:degraded:official_whitelist"],
                )
            return StepExecutionResult(
                ok=True,
                message="\n".join(read_result.value[:2]),
                context_blocks=read_result.value,
                tool_audit=["web_read:allowed:official_whitelist"],
            )
        if step_type == "general_skill":
            allowed, reason = self.gateway._guard_skill_request(query=subproblem.query, saved_skill_id=None)
            if not allowed:
                return StepExecutionResult(ok=False, message=f"技能执行被拦截：{reason}", tool_audit=[f"skill_exec:blocked:{reason}"])
            skill_result = self.deps.container.isolation.execute(
                "skill-service",
                lambda: self.gateway._invoke_skill(subproblem.query, request.session_id, None, fail_features),
            )
            if not skill_result.ok or not skill_result.value:
                return StepExecutionResult(ok=False, message=f"技能执行失败：{skill_result.error or 'unknown'}")
            return StepExecutionResult(
                ok=True,
                message=str(skill_result.value),
                context_blocks=[f"[skill] {skill_result.value}"],
                tool_audit=[f"skill_exec:allowed:{reason}"],
            )
        if step_type == "saved_skill":
            if not request.saved_skill_id:
                return StepExecutionResult(ok=False, message="当前没有可用的历史技能。")
            allowed, reason = self.gateway._guard_skill_request(query=subproblem.query, saved_skill_id=request.saved_skill_id)
            if not allowed:
                return StepExecutionResult(ok=False, message=f"历史技能调用被拦截：{reason}", tool_audit=[f"use_saved_skill:blocked:{reason}"])
            skill_result = self.deps.container.isolation.execute(
                "saved-skill-service",
                lambda: self.gateway._invoke_skill(subproblem.query, request.session_id, request.saved_skill_id, fail_features),
            )
            if not skill_result.ok or not skill_result.value:
                return StepExecutionResult(ok=False, message=f"历史技能执行失败：{skill_result.error or 'unknown'}")
            return StepExecutionResult(
                ok=True,
                message=str(skill_result.value),
                context_blocks=[f"[saved-skill] {skill_result.value}"],
                tool_audit=[f"use_saved_skill:allowed:{reason}"],
            )
        if step_type == "mcp_discover":
            runtime = self.get_mcp_runtime(trace_id)
            notes = list(runtime.notes)
            if runtime.tools:
                message = "\n".join(f"- {getattr(tool, 'name', 'unknown_tool')}" for tool in runtime.tools[:8])
                return StepExecutionResult(
                    ok=True,
                    message=message or "当前没有可用的 MCP 工具。",
                    tool_audit=[f"agent_tool:mcp_runtime:{','.join(item.alias for item in runtime.servers)}"],
                    notes=notes,
                )
            if runtime.servers:
                return StepExecutionResult(
                    ok=False,
                    message="MCP 服务已配置但工具不可用。",
                    tool_audit=[f"agent_tool:mcp_runtime_unavailable:{','.join(item.alias for item in runtime.servers)}"],
                    notes=notes,
                )
            return StepExecutionResult(ok=False, message="当前未配置 MCP 工具。", notes=notes)
        if step_type == "mcp_execute":
            runtime = self.get_mcp_runtime(trace_id)
            if not runtime.tools:
                return StepExecutionResult(ok=False, message="当前没有可执行的 MCP 工具。", notes=runtime.notes)
            tool = self._select_mcp_tool(runtime.tools, subproblem.query)
            with self._get_mcp_runtime_lock(trace_id):
                try:
                    result = asyncio.run(self._invoke_mcp_tool(tool, subproblem.query))
                except Exception as exc:
                    return StepExecutionResult(ok=False, message=f"MCP 工具执行失败：{exc.__class__.__name__}", notes=runtime.notes)
            return StepExecutionResult(
                ok=bool(str(result).strip()),
                message=str(result).strip() or "MCP 工具未返回内容。",
                context_blocks=[f"[mcp] {str(result).strip()}"] if str(result).strip() else [],
                tool_audit=[f"agent_tool:mcp_execute:{getattr(tool, 'name', 'unknown_tool')}"],
                notes=runtime.notes,
            )
        if step_type == "citation_guard":
            guard_result = self.deps.container.isolation.execute(
                "citation-guard",
                lambda: self.gateway._invoke_citation_guard(self.dedupe_sources(subproblem.sources, 5), fail_features),
            )
            if guard_result.ok and guard_result.value:
                return StepExecutionResult(ok=True, message="引用校验通过。")
            return StepExecutionResult(ok=False, message="引用校验失败或证据不足。")
        if step_type == "synthesize_step":
            chunks = [
                *(f"[step]{value}" for value in subproblem.step_outputs.values() if value.strip()),
                *(f"[note]{item}" for item in subproblem.notes if item.strip()),
            ]
            return StepExecutionResult(
                ok=bool(chunks),
                message="\n".join(chunks[:6]) if chunks else "当前步骤缺少可汇总内容。",
                context_blocks=chunks[:6],
            )
        return StepExecutionResult(ok=False, message=f"未支持的计划步骤：{step_type}")

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

    def review_step(self, step: PlanStep, result: StepExecutionResult) -> StepReviewResult:
        if not result.ok:
            return StepReviewResult(ok=False, message=result.message or "步骤执行未返回有效结果。")
        if step.step_type in {"local_rag_search", "official_web_search", "official_web_read"} and not result.context_blocks:
            return StepReviewResult(ok=False, message="步骤缺少可用证据。")
        if step.step_type == "citation_guard" and "通过" not in result.message:
            return StepReviewResult(ok=False, message=result.message or "引用校验失败。")
        if not (result.message or "").strip():
            return StepReviewResult(ok=False, message="步骤执行结果为空。")
        return StepReviewResult(ok=True, message="步骤满足要求。")

    def replan_subproblem(self, subproblem: SubproblemState, request: ChatRequest) -> SubproblemState:
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
        if degraded and status == "ok":
            status = "degraded"
        if output_flagged:
            tool_audit = [*tool_audit, f"safety_audit:output_sanitized:{output_reason}"]
            merged_text = audited_text
            status = "degraded"
        self.gateway._persist_memory_side_effects(request.session_id, last_user, merged_text)
        return SessionResult(
            session_id=request.session_id,
            trace_id=trace_id,
            text=merged_text,
            status=status,  # type: ignore[arg-type]
            degraded_features=list(dict.fromkeys(degraded)),
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

    def _rule_rewrite_query(self, *, last_user: str, memory_text: str, strategy: AgentStrategy) -> str:
        normalized = " ".join(last_user.split()).strip()
        if strategy == "quality" and memory_text and any(token in normalized for token in ("这个", "那个", "它", "那")):
            first_memory_line = next((line[2:] for line in memory_text.splitlines() if line.startswith("- ")), "")
            if first_memory_line:
                return f"{normalized}（结合上下文：{first_memory_line}）"
        return normalized

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

    def _should_use_mcp(self, *, query: str, route_label: str, strategy: AgentStrategy) -> bool:
        servers, _ = load_mcp_server_configs(self.deps.services.settings)
        if not servers:
            return False

        normalized = query.lower()
        if any(token in normalized for token in ("mcp", "外部工具", "模型上下文协议", "bing", "fetch")):
            return True

        capability_text = " ".join(
            " ".join(
                [
                    item.alias,
                    item.original_name,
                    item.command,
                    *item.args,
                    item.url,
                ]
            ).lower()
            for item in servers
        )
        has_search_capability = any(token in capability_text for token in ("search", "bing", "serp", "query"))
        has_fetch_capability = any(token in capability_text for token in ("fetch", "crawl", "read", "browser", "web"))
        if has_search_capability and (
            route_label == "time_sensitive" or any(token in query for token in ("搜索", "查询", "查一下", "搜一下", "最新", "公告", "官网", "官方"))
        ):
            return True
        if has_fetch_capability and any(token in query for token in ("网页", "页面", "链接", "抓取", "读取", "打开")):
            return True
        return strategy == "quality" and route_label == "time_sensitive" and (has_search_capability or has_fetch_capability)

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

    def _select_mcp_tool(self, tools: list[Any], query: str) -> Any:
        if len(tools) == 1:
            return tools[0]

        normalized_query = query.lower()
        query_prefers_search = any(token in query for token in ("搜索", "查询", "查一下", "搜一下", "最新", "公告", "官网", "官方"))
        query_prefers_fetch = any(token in query for token in ("网页", "页面", "链接", "抓取", "读取", "打开"))

        def _score(tool: Any) -> tuple[int, str]:
            name = str(getattr(tool, "name", "") or "").lower()
            description = str(getattr(tool, "description", "") or "").lower()
            haystack = f"{name} {description}"
            score = 0
            if "search" in normalized_query or query_prefers_search:
                if any(token in haystack for token in ("search", "bing", "query", "web_search")):
                    score += 3
            if query_prefers_fetch:
                if any(token in haystack for token in ("fetch", "read", "crawl", "browser", "web_read")):
                    score += 3
            if "mcp" in haystack:
                score += 1
            return score, name

        return max(tools, key=_score)
