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


DEFAULT_PROMPTS = [
    "请简短介绍招生政策重点",
    "再说一下学费和招生咨询电话",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="模拟前端通过真实 HTTP 调用 /api/chat 与 /api/chat/stream。")
    parser.add_argument("--base-url", default=os.getenv("BACKEND_API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--output", default="reports/frontend_api_chat_probe.json")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--session-id", default=uuid.uuid4().hex)
    parser.add_argument("--feature", action="append", dest="features", help="可重复传入 feature，默认 rag + citation_guard。")
    parser.add_argument("--prompt", action="append", dest="prompts", help="可重复传入多轮用户问题。")
    parser.add_argument("--model", default="zyit-pro")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--strict-citation", action="store_true", default=True)
    return parser


def parse_sse_body(body: str) -> dict[str, Any]:
    messages: list[str] = []
    done_payload: dict[str, Any] = {}
    current_event = ""
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.splitlines():
            if line.startswith("event: "):
                current_event = line.removeprefix("event: ").strip()
                continue
            if not line.startswith("data: "):
                continue
            payload = json.loads(line.removeprefix("data: ").strip())
            if current_event == "message":
                messages.append(str(payload.get("delta", "")))
            elif current_event == "done":
                done_payload = payload
    return {
        "text": "".join(messages),
        "done": done_payload,
    }


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
        "model": model,
    }
    client_logs: list[dict[str, Any]] = []

    create_started = time.perf_counter()
    create_response = client.post(f"{base_url}/api/chat", json=request_payload)
    create_latency_ms = round((time.perf_counter() - create_started) * 1000, 2)
    create_body = create_response.json() if create_response.headers.get("content-type", "").startswith("application/json") else {"raw": create_response.text}
    client_logs.append(
        {
            "stage": "create_chat",
            "status_code": create_response.status_code,
            "latency_ms": create_latency_ms,
        }
    )

    stream_started = time.perf_counter()
    stream_response = client.get(f"{base_url}/api/chat/stream", params={"session_id": session_id})
    stream_latency_ms = round((time.perf_counter() - stream_started) * 1000, 2)
    stream_body = stream_response.text
    parsed_stream = parse_sse_body(stream_body)
    client_logs.append(
        {
            "stage": "stream_chat",
            "status_code": stream_response.status_code,
            "latency_ms": stream_latency_ms,
        }
    )

    notes = [
        "本脚本通过真实 HTTP 请求模拟前端联调，不使用 TestClient。",
        "后端运行日志仍打印在后端服务控制台；本报告只记录客户端可观测结果和后端 SSE 回包。",
    ]
    if create_body.get("status") == "failed":
        notes.append("本轮生成阶段失败，请优先检查后端控制台日志、trace_id 和模型配置。")

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "request": request_payload,
        "create_response": {
            "status_code": create_response.status_code,
            "body": create_body,
        },
        "stream_response": {
            "status_code": stream_response.status_code,
            "text_preview": preview(parsed_stream.get("text", "")),
            "done": parsed_stream.get("done", {}),
            "raw_preview": preview(stream_body, limit=1200),
        },
        "client_logs": client_logs,
        "notes": notes,
    }


def main() -> int:
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
        ],
    }

    try:
        with httpx.Client(timeout=args.timeout) as client:
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
