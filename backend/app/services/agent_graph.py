from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import replace
from typing import Iterator

from langgraph.graph import END, StateGraph

from app.models import ChatRequest, SessionResult
from app.services.agent_runtime import AgentRuntime, StepSink
from app.services.agent_types import AgentGraphState, PlanStep, StepExecutionResult, SubproblemState


class AgentGraphRunner:
    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime
        self._graph = self._build_graph()
        self._last_stream_session: SessionResult | None = None

    def run_sync(
        self,
        *,
        request: ChatRequest,
        fail_features: set[str],
        step_sink: StepSink | None = None,
    ) -> SessionResult:
        trace_id = self.runtime.new_trace_id()
        last_user = self.runtime.get_last_user(request)
        input_blocked, input_reason, safe_reply = self.runtime.audit_user_input(last_user)
        step_events = []
        if input_blocked:
            self.runtime.emit_step(
                events=step_events,
                sink=step_sink,
                strategy=request.agent_strategy,
                node="input_audit",
                title="输入安全审查",
                status="degraded",
                message=input_reason,
            )
            return SessionResult(
                session_id=request.session_id,
                trace_id=trace_id,
                text=safe_reply,
                status="degraded",
                degraded_features=[],
                sources=[],
                tool_audit=[f"safety_audit:input_blocked:{input_reason}", "agent:blocked"],
                agent_strategy=request.agent_strategy,
                agent_trace=step_events,
            )

        initial_state: AgentGraphState = {
            "trace_id": trace_id,
            "session_id": request.session_id,
            "last_user": last_user,
            "request": request,
            "fail_features": fail_features,
            "agent_strategy": request.agent_strategy,
            "effective_features": list(request.features),
            "route_label": "policy",
            "route_reason": "default_policy",
            "memory_context": [],
            "rewritten_query": last_user,
            "subproblems": [],
            "current_subproblems": [],
            "subproblem_results": [],
            "final_text": "",
            "sources": [],
            "tool_audit": [],
            "notes": [],
            "degraded_features": [],
            "step_events": step_events,
            "failure_reason": None,
            "pending_retries": [],
            "generation_context_blocks": [],
            "generation_notes": [],
            "merge_summary": "",
            "blocked_reply": None,
            "blocked_audit": [],
            "current_round_complete": False,
            "status": "ok",
        }

        try:
            try:
                final_state = self._graph.invoke(
                    {
                        **initial_state,
                        "_step_sink": step_sink,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                return self.runtime.build_failure_session(request=request, exc=exc, step_events=step_events)

            if final_state.get("failure_reason") and not final_state.get("final_text"):
                return self.runtime.build_failure_session(
                    request=request,
                    exc=RuntimeError(str(final_state["failure_reason"])),
                    step_events=step_events,
                )

            return self.runtime.build_final_session(
                request=request,
                trace_id=trace_id,
                last_user=last_user,
                final_text=str(final_state["final_text"]),
                sources=list(final_state["sources"]),
                tool_audit=list(final_state["tool_audit"]),
                degraded_features=list(final_state["degraded_features"]),
                step_events=list(final_state["step_events"]),
                status_override=final_state.get("status"),
                error_message=final_state.get("failure_reason"),
            )
        finally:
            self.runtime.release_mcp_runtime(trace_id)

    def run_stream(
        self,
        *,
        request: ChatRequest,
        fail_features: set[str],
        text_chunker,
    ) -> Iterator[tuple[str, dict]]:
        event_queue: queue.Queue[object] = queue.Queue()
        sentinel = object()
        result_holder: dict[str, SessionResult] = {}
        error_holder: dict[str, BaseException] = {}

        def sink(event):
            event_queue.put(event)

        def worker() -> None:
            try:
                result_holder["session"] = self.run_sync(request=request, fail_features=fail_features, step_sink=sink)
            except BaseException as exc:  # noqa: BLE001
                error_holder["error"] = exc
            finally:
                event_queue.put(sentinel)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = event_queue.get()
            if item is sentinel:
                break
            yield "step", item.model_dump(mode="json")

        thread.join()
        if "error" in error_holder:
            raise error_holder["error"]
        session = result_holder["session"]
        self._last_stream_session = session
        for chunk in text_chunker(session.text):
            yield "message", chunk.data
        yield "done", {
            "finish_reason": session.finish_reason,
            "status": session.status,
            "degraded_features": session.degraded_features,
            "sources": [item.model_dump() for item in session.sources],
            "trace_id": session.trace_id,
            "tool_audit": session.tool_audit,
            "error_message": session.error_message,
            "agent_strategy": session.agent_strategy,
        }

    def _build_graph(self):
        graph = StateGraph(dict)
        graph.add_node("load_memory", self._load_memory)
        graph.add_node("preprocess_query", self._preprocess_query)
        graph.add_node("split_query", self._split_query)
        graph.add_node("build_subproblem_plan", self._build_subproblem_plan)
        graph.add_node("run_plan_step", self._run_plan_step)
        graph.add_node("review_step", self._review_step)
        graph.add_node("retry_or_escalate", self._retry_or_escalate)
        graph.add_node("merge_subproblem_results", self._merge_subproblem_results)
        graph.add_node("generate_final_answer", self._generate_final_answer)
        graph.add_node("postprocess_async_dispatch", self._postprocess_async_dispatch)
        graph.set_entry_point("load_memory")
        graph.add_edge("load_memory", "preprocess_query")
        graph.add_edge("preprocess_query", "split_query")
        graph.add_edge("split_query", "build_subproblem_plan")
        graph.add_edge("build_subproblem_plan", "run_plan_step")
        graph.add_edge("run_plan_step", "review_step")
        graph.add_conditional_edges(
            "review_step",
            lambda state: "retry_or_escalate" if state["pending_retries"] else "merge_subproblem_results",
        )
        graph.add_conditional_edges(
            "retry_or_escalate",
            lambda state: "run_plan_step" if state["current_subproblems"] else "merge_subproblem_results",
        )
        graph.add_edge("merge_subproblem_results", "generate_final_answer")
        graph.add_edge("generate_final_answer", "postprocess_async_dispatch")
        graph.add_edge("postprocess_async_dispatch", END)
        return graph.compile()

    def _load_memory(self, state: dict) -> dict:
        sink = state.get("_step_sink")
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="load_memory",
            title="加载会话记忆",
            status="started",
        )
        context_blocks, memory_text, notes = self.runtime.load_memory_context(state["session_id"])
        state["memory_context"] = context_blocks
        state["notes"].extend(notes)
        state["tool_audit"].append("agent_tool:memory_recall")
        state["notes"].append(memory_text if memory_text.strip() else "当前没有可用记忆。")
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="load_memory",
            title="加载会话记忆",
            status="completed",
            message=f"已装载 {len(context_blocks)} 条上下文片段",
        )
        return state

    def _preprocess_query(self, state: dict) -> dict:
        sink = state.get("_step_sink")
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="preprocess_query",
            title="前处理与路由",
            status="started",
        )
        route_decision = self.runtime.route_features(state["last_user"], state["request"])
        state["effective_features"] = route_decision.features
        state["route_label"] = route_decision.route_label
        state["route_reason"] = route_decision.reason
        state["tool_audit"].extend(route_decision.audit)
        state["notes"].extend(route_decision.notes)
        memory_text = next((item for item in state["notes"] if "记忆" in item), "")
        state["rewritten_query"] = self.runtime.rewrite_query(
            request=state["request"],
            last_user=state["last_user"],
            memory_text=memory_text,
            strategy=state["agent_strategy"],
        )
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="preprocess_query",
            title="前处理与路由",
            status="completed",
            message=f"{route_decision.route_label}:{route_decision.reason}",
        )
        return state

    def _split_query(self, state: dict) -> dict:
        sink = state.get("_step_sink")
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="split_query",
            title="拆分子问题",
            status="started",
        )
        subproblems = self.runtime.split_query(
            state["rewritten_query"],
            state["agent_strategy"],
            state["request"],
        )
        if not subproblems:
            subproblems = [state["rewritten_query"]]
        state["subproblems"] = [
            SubproblemState(subproblem_id=f"sp-{idx + 1}", query=item)
            for idx, item in enumerate(subproblems)
        ]
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="split_query",
            title="拆分子问题",
            status="completed",
            message=f"共 {len(state['subproblems'])} 个子问题",
        )
        return state

    def _build_subproblem_plan(self, state: dict) -> dict:
        sink = state.get("_step_sink")
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="build_subproblem_plan",
            title="构建子问题计划",
            status="started",
        )
        built: list[SubproblemState] = []
        for subproblem in state["subproblems"]:
            subproblem.plan_steps = self.runtime.build_plan(
                query=subproblem.query,
                effective_features=state["effective_features"],
                route_label=state["route_label"],
                request=state["request"],
                strategy=state["agent_strategy"],
            )
            built.append(subproblem)
        state["subproblems"] = built
        state["current_subproblems"] = built
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="build_subproblem_plan",
            title="构建子问题计划",
            status="completed",
            message="已为所有子问题生成执行计划",
        )
        return state

    def _run_plan_step(self, state: dict) -> dict:
        sink = state.get("_step_sink")
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="run_plan_step",
            title="执行计划步骤",
            status="started",
            message=f"并行子问题数：{len(state['current_subproblems'])}",
        )
        processed = asyncio.run(
            self._process_subproblems(
                subproblems=list(state["current_subproblems"]),
                state=state,
                sink=sink,
            )
        )
        result_map = {item.subproblem_id: item for item in state["subproblem_results"]}
        for item in processed:
            result_map[item.subproblem_id] = item
        state["subproblem_results"] = list(result_map.values())
        state["current_round_complete"] = True
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="run_plan_step",
            title="执行计划步骤",
            status="completed",
            message="当前轮执行完成",
        )
        return state

    async def _process_subproblems(self, *, subproblems: list[SubproblemState], state: dict, sink) -> list[SubproblemState]:
        return await asyncio.gather(
            *[
                asyncio.to_thread(self._execute_subproblem, subproblem, state, sink)
                for subproblem in subproblems
            ]
        )

    def _execute_subproblem(self, subproblem: SubproblemState, state: dict, sink) -> SubproblemState:
        max_retry = 1 if state["agent_strategy"] == "speed" else 2
        working = replace(subproblem)
        working.plan_steps = list(subproblem.plan_steps)
        working.step_outputs = dict(subproblem.step_outputs)
        working.notes = list(subproblem.notes)
        working.context_blocks = list(subproblem.context_blocks)
        working.sources = list(subproblem.sources)
        working.tool_audit = list(subproblem.tool_audit)
        working.web_hits = list(subproblem.web_hits)
        working.status = "pending"
        working.degraded = False

        for idx, step in enumerate(working.plan_steps, start=1):
            attempt = 0
            while True:
                self.runtime.emit_step(
                    events=state["step_events"],
                    sink=sink,
                    strategy=state["agent_strategy"],
                    node=step.step_type,
                    title=step.title,
                    status="started",
                    message=step.instruction or None,
                    subproblem_id=working.subproblem_id,
                    plan_step_index=idx,
                    attempt=attempt + 1,
                )
                try:
                    result = self.runtime.run_subproblem_agent(
                        step=step,
                        subproblem=working,
                        request=state["request"],
                        fail_features=state["fail_features"],
                        effective_features=state["effective_features"],
                        memory_context_blocks=state["memory_context"],
                        trace_id=state["trace_id"],
                        route_label=state["route_label"],
                        step_events=state["step_events"],
                        sink=sink,
                        attempt=attempt + 1,
                    )
                except Exception as exc:  # noqa: BLE001
                    result = StepExecutionResult(ok=False, message=f"计划节点执行失败：{exc.__class__.__name__}: {exc}", notes=[f"计划节点执行失败：{exc.__class__.__name__}: {exc}"])
                review = self.runtime.review_step(step, result)
                working.tool_audit.extend(result.tool_audit)
                working.notes.extend(result.notes)
                working.context_blocks.extend(result.context_blocks)
                working.sources = self.runtime.dedupe_sources([*working.sources, *result.sources], limit=5)
                if result.web_hits:
                    working.web_hits = result.web_hits
                if review.ok:
                    working.step_outputs[f"{idx}:{step.step_type}"] = result.message
                    working.current_step_index = idx
                    self.runtime.emit_step(
                        events=state["step_events"],
                        sink=sink,
                        strategy=state["agent_strategy"],
                        node=step.step_type,
                        title=step.title,
                        status="completed",
                        message=review.message,
                        subproblem_id=working.subproblem_id,
                        plan_step_index=idx,
                        attempt=attempt + 1,
                    )
                    break
                if attempt < max_retry:
                    attempt += 1
                    working.attempt_count += 1
                    self.runtime.emit_step(
                        events=state["step_events"],
                        sink=sink,
                        strategy=state["agent_strategy"],
                        node=step.step_type,
                        title=step.title,
                        status="retrying",
                        message=review.message,
                        subproblem_id=working.subproblem_id,
                        plan_step_index=idx,
                        attempt=attempt,
                    )
                    continue
                if state["agent_strategy"] == "quality" and working.replan_count < 1:
                    working.status = "needs_replan"
                    working.notes.append(review.message)
                    self.runtime.emit_step(
                        events=state["step_events"],
                        sink=sink,
                        strategy=state["agent_strategy"],
                        node=step.step_type,
                        title=step.title,
                        status="retrying",
                        message=f"准备重规划：{review.message}",
                        subproblem_id=working.subproblem_id,
                        plan_step_index=idx,
                        attempt=attempt + 1,
                    )
                    return working
                if state["agent_strategy"] == "speed":
                    working.degraded = True
                    working.notes.append(review.message)
                    working.tool_audit.append(f"step:degraded:{step.step_type}")
                    self.runtime.emit_step(
                        events=state["step_events"],
                        sink=sink,
                        strategy=state["agent_strategy"],
                        node=step.step_type,
                        title=step.title,
                        status="degraded",
                        message=review.message,
                        subproblem_id=working.subproblem_id,
                        plan_step_index=idx,
                        attempt=attempt + 1,
                    )
                    break
                working.status = "failed"
                working.notes.append(review.message)
                self.runtime.emit_step(
                    events=state["step_events"],
                    sink=sink,
                    strategy=state["agent_strategy"],
                    node=step.step_type,
                    title=step.title,
                    status="failed",
                    message=review.message,
                    subproblem_id=working.subproblem_id,
                    plan_step_index=idx,
                    attempt=attempt + 1,
                )
                return working

        if working.status == "pending":
            working.status = "degraded" if working.degraded else "completed"
        return working

    def _review_step(self, state: dict) -> dict:
        sink = state.get("_step_sink")
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="review_step",
            title="审查步骤结果",
            status="started",
        )
        state["pending_retries"] = [item for item in state["subproblem_results"] if item.status == "needs_replan"]
        if any(item.status == "failed" for item in state["subproblem_results"]):
            state["status"] = "degraded"
            state["failure_reason"] = "部分子问题执行失败"
        elif any(item.status == "degraded" for item in state["subproblem_results"]):
            state["status"] = "degraded"
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="review_step",
            title="审查步骤结果",
            status="completed",
            message=f"待重规划子问题：{len(state['pending_retries'])}",
        )
        return state

    def _retry_or_escalate(self, state: dict) -> dict:
        sink = state.get("_step_sink")
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="retry_or_escalate",
            title="重试或重规划",
            status="started",
        )
        replanned: list[SubproblemState] = []
        for item in state["pending_retries"]:
            new_item = self.runtime.replan_subproblem(item, state["request"])
            replanned.append(new_item)
        state["current_subproblems"] = replanned
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="retry_or_escalate",
            title="重试或重规划",
            status="completed",
            message=f"已刷新 {len(replanned)} 个子问题的候选工具建议",
        )
        return state

    def _merge_subproblem_results(self, state: dict) -> dict:
        sink = state.get("_step_sink")
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="merge_subproblem_results",
            title="汇总子问题结果",
            status="started",
        )
        notes: list[str] = []
        context_blocks: list[str] = []
        sources = []
        degraded_features = list(state["degraded_features"])
        for item in state["subproblem_results"]:
            notes.append(f"[{item.subproblem_id}][{item.status}] {item.query}")
            notes.extend(item.notes[:4])
            context_blocks.extend(item.context_blocks[:6])
            sources.extend(item.sources)
            state["tool_audit"].extend(item.tool_audit)
            if item.status in {"degraded", "failed"}:
                degraded_features.append("citation_guard" if not item.sources else "rag")
        state["sources"] = self.runtime.dedupe_sources(sources, limit=5)
        state["notes"].extend(notes)
        state["generation_context_blocks"] = context_blocks[:12]
        state["merge_summary"] = "\n".join(notes[:10])
        state["degraded_features"] = list(dict.fromkeys(degraded_features))
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="merge_subproblem_results",
            title="汇总子问题结果",
            status="completed",
            message=f"汇总 {len(state['subproblem_results'])} 个子问题",
        )
        return state

    def _generate_final_answer(self, state: dict) -> dict:
        sink = state.get("_step_sink")
        generation_context_blocks = self.runtime.compact_generation_context(
            context_blocks=state["generation_context_blocks"] or state["memory_context"],
            strategy=state["agent_strategy"],
        )
        generation_notes = self.runtime.compact_feature_notes(
            notes=state["notes"],
            strategy=state["agent_strategy"],
        )
        state["generation_context_blocks"] = generation_context_blocks
        state["generation_notes"] = generation_notes
        context_chars = sum(len(item) for item in generation_context_blocks)
        note_chars = sum(len(item) for item in generation_notes)
        state["tool_audit"].append(
            "generation:prepared:"
            f"context_blocks={len(generation_context_blocks)}:"
            f"context_chars={context_chars}:"
            f"notes={len(generation_notes)}:"
            f"note_chars={note_chars}"
        )
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="generate_final_answer",
            title="生成最终回答",
            status="started",
            message=f"上下文 {len(generation_context_blocks)} 条，备注 {len(generation_notes)} 条",
        )
        generation_result = self.runtime.deps.container.isolation.execute(
            "generation-service",
            lambda: self.runtime.gateway._invoke_generation(
                user_query=state["last_user"],
                context_blocks=generation_context_blocks,
                feature_notes=generation_notes,
                request=state["request"],
                fail_features=state["fail_features"],
            ),
        )
        if not generation_result.ok or generation_result.value is None:
            error_message = generation_result.error or "generation failed"
            state["tool_audit"].append(f"generation:error:{error_message}")
            if self.runtime.should_degrade_generation_error(error_message):
                state["final_text"] = self.runtime.build_rule_based_final_answer(
                    query=state["last_user"],
                    context_blocks=generation_context_blocks,
                    notes=generation_notes,
                    error_message=error_message,
                )
                state["status"] = "degraded"
                state["failure_reason"] = error_message
                state["tool_audit"].append("generation:fallback:rule_based")
                self.runtime.emit_step(
                    events=state["step_events"],
                    sink=sink,
                    strategy=state["agent_strategy"],
                    node="generate_final_answer",
                    title="生成最终回答",
                    status="degraded",
                    message=f"最终生成超时，已切换保守汇总：{error_message}",
                )
                return state
            state["status"] = "failed"
            state["failure_reason"] = error_message
            self.runtime.emit_step(
                events=state["step_events"],
                sink=sink,
                strategy=state["agent_strategy"],
                node="generate_final_answer",
                title="生成最终回答",
                status="failed",
                message=error_message,
            )
            raise RuntimeError(error_message)
        generation_output = generation_result.value
        state["final_text"] = generation_output.text
        state["tool_audit"].append(
            "generation:"
            f"{generation_output.route}:"
            f"{generation_output.model or 'unknown'}:"
            f"cache_{'hit' if generation_output.cache_hit else 'miss'}"
        )
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="generate_final_answer",
            title="生成最终回答",
            status="completed",
            message=f"最终回答生成完成，输出长度 {len(state['final_text'])} 字符",
        )
        return state

    def _postprocess_async_dispatch(self, state: dict) -> dict:
        sink = state.get("_step_sink")
        self.runtime.emit_step(
            events=state["step_events"],
            sink=sink,
            strategy=state["agent_strategy"],
            node="postprocess_async_dispatch",
            title="后处理",
            status="completed",
            message="将于会话落库后异步写入记忆",
        )
        return state
