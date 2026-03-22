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
    parser = argparse.ArgumentParser(description="模拟前端通过真实 HTTP 调用 /api/chat/stream。")
    parser.add_argument("--base-url", default=os.getenv("BACKEND_API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--output", default=os.getenv("PROBE_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("PROBE_TIMEOUT_SECONDS", "30.0")))
    parser.add_argument("--session-id", default=uuid.uuid4().hex)
    parser.add_argument("--feature", action="append", dest="features", help="可重复传入 feature，默认 rag + citation_guard。")
    parser.add_argument("--prompt", action="append", dest="prompts", help="可重复传入多轮用户问题。")
    parser.add_argument("--model", default=os.getenv("PROBE_MODEL", "").strip(), help="可选：显式指定模型；默认留空，让后端自行路由。")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("PROBE_TEMPERATURE", "0.2")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("PROBE_TOP_P", "0.85")))
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


def healthcheck(client: httpx.Client, base_url: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    response = client.get(f"{base_url}/healthz/dependencies")
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return {
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
    }


def run_round(
    client: httpx.Client,
    *,
    base_url: str,
    session_id: str,
    prompt: str,
    features: list[str],
    model: str,
    temperature: float,
    top_p: float,
    strict_citation: bool,
) -> dict[str, Any]:
    request_payload = {
        "session_id": session_id,
        "messages": [{"role": "user", "content": prompt}],
        "mode": "chat",
        "stream": True,
        "features": features,
        "strict_citation": strict_citation,
        "temperature": temperature,
        "top_p": top_p,
    }
    if model.strip():
        request_payload["model"] = model.strip()
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

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "request": request_payload,
        "stream_response": {
            "status_code": 200,
            "text_preview": preview("".join(messages)),
            "done": done_payload,
            "delta_logs": delta_logs[:40],
            "raw_preview": preview(" ".join(raw_chunks), limit=1200),
        },
        "client_logs": client_logs,
        "notes": notes,
    }


def main() -> int:
    env_path = bootstrap_env()
    args = build_parser().parse_args()
    base_url = str(args.base_url).rstrip("/")
    features = args.features or ["rag", "citation_guard"]
    prompts = args.prompts or list(DEFAULT_PROMPTS)
    session_id = str(args.session_id)
    output_path = Path(args.output)

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "session_id": session_id,
        "features": features,
        "prompts": prompts,
        "healthcheck": None,
        "rounds": [],
        "notes": [
            "如果 healthcheck 或请求失败，优先确认后端是否已通过 python 启动并监听 8000 端口。",
            "推荐先在另一个终端运行: cd backend && python main.py",
            f"dotenv 已加载: {env_path}",
        ],
    }

    try:
        with httpx.Client(timeout=args.timeout, trust_env=False) as client:
            report["healthcheck"] = healthcheck(client, base_url)
            for prompt in prompts:
                report["rounds"].append(
                    run_round(
                        client,
                        base_url=base_url,
                        session_id=session_id,
                        prompt=prompt,
                        features=features,
                        model=args.model,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        strict_citation=args.strict_citation,
                    )
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
