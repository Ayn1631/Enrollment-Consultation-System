from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT = BACKEND_DIR / "reports" / "llm_api_test.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="直接使用 openai 官方库测试 LLM API，不走 pytest。")
    parser.add_argument("--prompt", default="请用一句话回答：中原工学院理工科专业学费一般是多少？最后输出hello world/")
    parser.add_argument("--system", default="你是一个简洁的招生咨询助手，只回答业务问题。")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.85)
    return parser


def load_runtime_config() -> dict[str, str]:
    env_path = BACKEND_DIR / ".env"
    load_dotenv(env_path, override=False)

    endpoint = (os.getenv("LLM_API_URL") or os.getenv("API_URL") or "").strip()
    api_key = (os.getenv("LLM_API_KEY") or os.getenv("API_KEY") or "").strip()
    model = (os.getenv("GENERATION_LIGHT_MODEL") or "gpt-5.4-mini").strip()
    return {
        "env_path": str(env_path),
        "endpoint": endpoint,
        "api_key": api_key,
        "model": model,
    }


def resolve_base_url(endpoint: str) -> str:
    for suffix in ("/chat/completions", "/responses", "/completions"):
        if endpoint.endswith(suffix):
            return endpoint[: -len(suffix)]
    return endpoint.rstrip("/")


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    runtime = load_runtime_config()
    output_path = Path(args.output)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "env_path": runtime["env_path"],
        "endpoint": runtime["endpoint"],
        "base_url": resolve_base_url(runtime["endpoint"]) if runtime["endpoint"] else "",
        "model": runtime["model"],
        "prompt": args.prompt,
        "system": args.system,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    if not runtime["endpoint"]:
        report["error"] = {"type": "ConfigError", "message": "未读取到 LLM_API_URL 或 API_URL。"}
        write_report(output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    if not runtime["api_key"]:
        report["error"] = {"type": "ConfigError", "message": "未读取到 LLM_API_KEY 或 API_KEY。"}
        write_report(output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    try:
        client = OpenAI(
            api_key=runtime["api_key"],
            base_url=resolve_base_url(runtime["endpoint"]),
            timeout=args.timeout,
            max_retries=0,
        )
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=runtime["model"],
            messages=[
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt},
            ],
            temperature=args.temperature,
            top_p=args.top_p,
            stream=False,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        answer = response.choices[0].message.content if response.choices else ""
        report["result"] = {
            "latency_ms": latency_ms,
            "id": response.id,
            "response_model": getattr(response, "model", runtime["model"]),
            "usage": response.usage.model_dump() if getattr(response, "usage", None) else {},
            "answer": answer,
        }
        client.close()
    except Exception as exc:  # noqa: BLE001
        report["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
        write_report(output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    write_report(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
