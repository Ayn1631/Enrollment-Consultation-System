from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

import httpx
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_PATH = BACKEND_DIR / "reports" / "frontend_api_chat_probe.json"
DEFAULT_PROMPTS = ["理工科的学费是多少?"]


def bootstrap_env() -> Path:
    env_path = BACKEND_DIR / ".env"
    load_dotenv(env_path, override=False)
    return env_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="模拟前端通过真实 HTTP 调用后端关键功能接口。")
    parser.add_argument("--base-url", default=os.getenv("BACKEND_API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--output", default=os.getenv("PROBE_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("PROBE_TIMEOUT_SECONDS", "30.0")))
    parser.add_argument("--session-id", default=uuid.uuid4().hex)
    parser.add_argument("--feature", action="append", dest="features", help="可重复传入 feature，默认 rag + citation_guard。")
    parser.add_argument("--prompt", action="append", dest="prompts", help="可重复传入多轮用户问题。")
    parser.add_argument(
        "--mode",
        choices=["chat", "agent", "plan", "guide"],
        default=os.getenv("PROBE_MODE", "chat"),
        help="请求模式，默认 chat。",
    )
    parser.add_argument("--model", default=os.getenv("PROBE_MODEL", "").strip(), help="可选：显式指定模型；默认留空，让后端自行路由。")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("PROBE_TEMPERATURE", "0.2")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("PROBE_TOP_P", "0.85")))
    parser.add_argument("--saved-skill-id", default=os.getenv("PROBE_SAVED_SKILL_ID", "").strip())
    parser.add_argument("--session-title", default=os.getenv("PROBE_SESSION_TITLE", "脚本联调测试"))
    parser.add_argument("--skip-meta-checks", action="store_true", help="跳过 features/saved skills/mcp tools 接口检查。")
    parser.add_argument("--compress-after", action="store_true", help="在多轮对话结束后调用 /api/memory/compress。")
    parser.add_argument("--strict-citation", action="store_true", default=True)
    return parser


def parse_sse_block(block: str) -> dict[str, str] | None:
    lines = [line.rstrip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None
    event = "message"
    data_lines: list[str] = []
    for line in lines:
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    if not data_lines:
        return None
    return {"event": event, "data": "\n".join(data_lines)}


def iter_sse_events(response: httpx.Response):
    buffer = ""
    for chunk in response.iter_text():
        if not chunk:
            continue
        buffer += chunk.replace("\r\n", "\n")
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            parsed = parse_sse_block(block)
            if parsed is not None:
                yield parsed
    if buffer.strip():
        parsed = parse_sse_block(buffer.strip())
        if parsed is not None:
            yield parsed


def preview(text: str, limit: int = 800) -> str:
    compact = " ".join(text.split())
    return compact[:limit]


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_json_response(response: httpx.Response) -> Any:
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        return response.json()
    return response.text


def healthcheck(client: httpx.Client, base_url: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    response = client.get(f"{base_url}/healthz/dependencies")
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return {
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "body": safe_json_response(response),
    }


def probe_meta_endpoints(client: httpx.Client, base_url: str) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for label, path in (
        ("features", "/api/features"),
        ("saved_skills", "/api/skills/saved"),
        ("mcp_tools", "/api/mcp/tools"),
    ):
        started_at = time.perf_counter()
        response = client.get(f"{base_url}{path}")
        checks[label] = {
            "path": path,
            "status_code": response.status_code,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "body": safe_json_response(response),
        }
    return checks


def run_round(
    client: httpx.Client,
    *,
    base_url: str,
    session_id: str,
    mode: str,
    prompt: str,
    history_messages: list[dict[str, str]],
    features: list[str],
    model: str,
    temperature: float,
    top_p: float,
    strict_citation: bool,
    saved_skill_id: str,
) -> dict[str, Any]:
    request_messages = [*history_messages, {"role": "user", "content": prompt}]
    request_payload = {
        "session_id": session_id,
        "messages": request_messages,
        "mode": mode,
        "stream": True,
        "features": features,
        "strict_citation": strict_citation,
        "temperature": temperature,
        "top_p": top_p,
    }
    if model.strip():
        request_payload["model"] = model.strip()
    if saved_skill_id.strip():
        request_payload["saved_skill_id"] = saved_skill_id.strip()
    client_logs: list[dict[str, Any]] = []
    raw_chunks: list[str] = []
    messages: list[str] = []
    done_payload: dict[str, Any] = {}
    delta_logs: list[dict[str, Any]] = []
    first_event_ms: float | None = None
    first_delta_ms: float | None = None

    stream_started = time.perf_counter()
    with client.stream(
        "POST",
        f"{base_url}/api/chat/stream",
        json=request_payload,
        headers={"Accept": "text/event-stream"},
    ) as stream_response:
        stream_open_latency_ms = round((time.perf_counter() - stream_started) * 1000, 2)
        client_logs.append(
            {
                "stage": "open_stream",
                "status_code": stream_response.status_code,
                "latency_ms": stream_open_latency_ms,
            }
        )
        for event in iter_sse_events(stream_response):
            elapsed_ms = round((time.perf_counter() - stream_started) * 1000, 2)
            raw_chunks.append(f"event: {event['event']} data: {event['data']}")
            if first_event_ms is None:
                first_event_ms = elapsed_ms
            payload = json.loads(event["data"])
            if event["event"] == "message":
                delta = str(payload.get("delta", ""))
                if delta:
                    messages.append(delta)
                    if first_delta_ms is None:
                        first_delta_ms = elapsed_ms
                    delta_logs.append(
                        {
                            "elapsed_ms": elapsed_ms,
                            "delta_preview": preview(delta, limit=120),
                            "delta_length": len(delta),
                        }
                    )
            elif event["event"] == "done":
                done_payload = payload
    stream_total_latency_ms = round((time.perf_counter() - stream_started) * 1000, 2)
    client_logs.append(
        {
            "stage": "stream_done",
            "status_code": 200,
            "latency_ms": stream_total_latency_ms,
            "first_event_ms": first_event_ms,
            "first_delta_ms": first_delta_ms,
        }
    )

    notes = [
        "本脚本通过真实 HTTP 请求直接调用 POST /api/chat/stream，不使用 TestClient。",
        "后端运行日志仍打印在后端服务控制台；本报告记录首包延迟、delta 片段与 done 事件。",
    ]
    if done_payload.get("status") == "failed":
        notes.append("本轮生成阶段失败，请优先检查后端控制台日志、trace_id 和模型配置。")

    assistant_text = "".join(messages)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "request": request_payload,
        "stream_response": {
            "status_code": 200,
            "text_preview": preview(assistant_text),
            "done": done_payload,
            "delta_logs": delta_logs[:40],
            "raw_preview": preview(" ".join(raw_chunks), limit=1200),
        },
        "client_logs": client_logs,
        "notes": notes,
        "assistant_message": assistant_text,
    }


def compress_context(
    client: httpx.Client,
    *,
    base_url: str,
    session_id: str,
    session_title: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    payload = {
        "session_id": session_id,
        "session_title": session_title,
        "messages": messages,
    }
    started_at = time.perf_counter()
    response = client.post(f"{base_url}/api/memory/compress", json=payload)
    return {
        "request": payload,
        "status_code": response.status_code,
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "body": safe_json_response(response),
    }


def main() -> int:
    env_path = bootstrap_env()
    args = build_parser().parse_args()
    base_url = str(args.base_url).rstrip("/")
    features = args.features or ["rag", "citation_guard"]
    prompts = args.prompts or list(DEFAULT_PROMPTS)
    session_id = str(args.session_id)
    output_path = Path(args.output)
    conversation_messages: list[dict[str, str]] = []

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "session_id": session_id,
        "mode": args.mode,
        "features": features,
        "prompts": prompts,
        "healthcheck": None,
        "meta_checks": None,
        "rounds": [],
        "memory_compression": None,
        "notes": [
            "如果 healthcheck 或请求失败，优先确认后端是否已通过 python 启动并监听 8000 端口。",
            "推荐先在另一个终端运行: cd backend && python main.py",
            f"dotenv 已加载: {env_path}",
        ],
    }

    try:
        with httpx.Client(timeout=args.timeout, trust_env=False) as client:
            report["healthcheck"] = healthcheck(client, base_url)
            if not args.skip_meta_checks:
                report["meta_checks"] = probe_meta_endpoints(client, base_url)
            for prompt in prompts:
                round_result = run_round(
                    client,
                    base_url=base_url,
                    session_id=session_id,
                    mode=args.mode,
                    prompt=prompt,
                    history_messages=conversation_messages,
                    features=features,
                    model=args.model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    strict_citation=args.strict_citation,
                    saved_skill_id=args.saved_skill_id,
                )
                report["rounds"].append(round_result)
                conversation_messages.append({"role": "user", "content": prompt})
                assistant_text = str(round_result.get("assistant_message", "")).strip()
                if assistant_text:
                    conversation_messages.append({"role": "assistant", "content": assistant_text})
            if args.compress_after and conversation_messages:
                report["memory_compression"] = compress_context(
                    client,
                    base_url=base_url,
                    session_id=session_id,
                    session_title=args.session_title,
                    messages=conversation_messages,
                )
    except Exception as exc:  # noqa: BLE001
        report["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
        write_report(output_path, report)
        print(f"[probe_api_chat] 请求失败，报告已写入: {output_path}")
        print(json.dumps(report["error"], ensure_ascii=False, indent=2))
        return 1

    write_report(output_path, report)
    print(f"[probe_api_chat] 完成，报告已写入: {output_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
