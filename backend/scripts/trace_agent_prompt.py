from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any
import uuid

import httpx


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT = BACKEND_DIR / "reports" / "agent_prompt_trace.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="针对单个 prompt 追踪 agent 模式接口链路。")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default="我高考350分, 应该报什么专业?")
    parser.add_argument("--mode", default="agent", choices=["chat", "plan", "guide", "agent"])
    parser.add_argument("--strategy", default="quality", choices=["speed", "quality"])
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
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


def build_request(prompt: str, mode: str, strategy: str) -> dict[str, Any]:
    return {
        "session_id": uuid.uuid4().hex,
        "messages": [{"role": "user", "content": prompt}],
        "mode": mode,
        "stream": True,
        "features": ["rag", "citation_guard"],
        "strict_citation": True,
        "agent_strategy": strategy,
    }


def summarize_events(steps: list[dict[str, Any]], done: dict[str, Any]) -> dict[str, Any]:
    failed_steps = [item for item in steps if item.get("status") == "failed"]
    degraded_steps = [item for item in steps if item.get("status") == "degraded"]
    retry_steps = [item for item in steps if item.get("status") == "retrying"]
    last_meaningful_step = steps[-1] if steps else None
    return {
        "step_count": len(steps),
        "failed_steps": failed_steps,
        "degraded_steps": degraded_steps,
        "retry_steps": retry_steps,
        "last_step": last_meaningful_step,
        "done_status": done.get("status"),
        "done_error_message": done.get("error_message"),
        "done_trace_id": done.get("trace_id"),
        "done_tool_audit": done.get("tool_audit", []),
    }


def main() -> int:
    args = build_parser().parse_args()
    request_payload = build_request(args.prompt, args.mode, args.strategy)
    base_url = args.base_url.rstrip("/")
    output_path = Path(args.output)

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "request": request_payload,
        "create_chat": None,
        "stream_replay": None,
        "summary": None,
    }

    try:
        with httpx.Client(timeout=args.timeout, trust_env=False) as client:
            create_res = client.post(f"{base_url}/api/chat", json=request_payload)
            create_body = create_res.json()
            report["create_chat"] = {
                "status_code": create_res.status_code,
                "body": create_body,
            }

            session_id = str(create_body["session_id"])
            steps: list[dict[str, Any]] = []
            messages: list[str] = []
            done_payload: dict[str, Any] = {}

            with client.stream("GET", f"{base_url}/api/chat/stream", params={"session_id": session_id}) as stream_res:
                raw_events: list[dict[str, Any]] = []
                for item in iter_sse_events(stream_res):
                    payload = json.loads(item["data"])
                    raw_events.append({"event": item["event"], "payload": payload})
                    if item["event"] == "step":
                        steps.append(payload)
                    elif item["event"] == "message":
                        messages.append(str(payload.get("delta", "")))
                    elif item["event"] == "done":
                        done_payload = payload
                report["stream_replay"] = {
                    "status_code": stream_res.status_code,
                    "steps": steps,
                    "text": "".join(messages),
                    "done": done_payload,
                    "raw_events": raw_events,
                }

            report["summary"] = summarize_events(steps, done_payload)
    except Exception as exc:  # noqa: BLE001
        report["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"\n完整报告已写入: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
