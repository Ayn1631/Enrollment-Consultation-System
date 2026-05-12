from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import uuid
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from tqdm.auto import tqdm

'''
进行压测
'''


ROOT = Path(__file__).resolve().parents[1]  # 项目根目录
DEFAULT_CASES_FILE = ROOT / "tests" / "fixtures" / "perf_chat_cases.json"  # 默认问题集文件，实际使用时可通过命令行参数覆盖；建议准备至少 12 条问题，以保证测试过程中缓存和场景分布的多样性，从而更准确地反映性能表现
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "perf"  # 默认输出目录，实际使用时会在其下创建一个以当前时间命名的子目录来存放本轮测试结果
DEFAULT_CONCURRENCY = (5, 10, 15, 20)  # 预设并发档位，实际使用时可通过命令行参数覆盖
DEFAULT_REQUESTS_PER_STAGE_MULTIPLIER = 1.25  # 每档正式请求数 = ceil(concurrency * multiplier)
EVENT_PREVIEW_LIMIT = 120  # 长文本预览限制，过长会导致日志和报告难以阅读
MODEL_CONFIG_ERROR_PATTERN = re.compile(
    r"(未配置模型|模型.*不存在|invalid model|model.*not found|unsupported model|unknown model)",
    flags=re.IGNORECASE,
)

Mode = Literal["rag"]


@dataclass(frozen=True)
class PerfCase:
    case_id: str
    category: str
    query: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 /api/chat/stream 执行预检与并发压测。")
    parser.add_argument("--base-url", required=True, help="后端服务地址，例如 http://127.0.0.1:8000")
    parser.add_argument(
        "--mode",
        default="rag",
        choices=["rag"],
        help="压测模式；当前仅保留 rag 主链路。",
    )
    parser.add_argument(
        "--concurrency",
        default="5, 10, 15, 20, 25, 30, 45, 60, 75, 100",
        help="并发档位，逗号分隔，例如 5,10,15,20",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="每档预热请求数，不计入正式结果。",
    )
    parser.add_argument(
        "--requests-per-stage",
        type=int,
        default=0,
        help="每档正式请求数；传 0 时按并发档位和 multiplier 自动计算。",
    )
    parser.add_argument(
        "--requests-per-stage-multiplier",
        type=float,
        default=DEFAULT_REQUESTS_PER_STAGE_MULTIPLIER,
        help="当 --requests-per-stage=0 时，每档正式请求数按 ceil(concurrency * multiplier) 计算。",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=30.0,
        help="档位之间冷却秒数。",
    )
    parser.add_argument(
        "--precheck-stream-requests",
        type=int,
        default=0,
        help="每个模式的流式预检次数。",
    )
    parser.add_argument(
        "--precheck-sync-requests",
        type=int,
        default=0,
        help="每个模式的同步预检次数。",
    )
    parser.add_argument(
        "--cases-file",
        default=str(DEFAULT_CASES_FILE),
        help="固定问题集 JSON 文件路径。",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录；脚本会在其下创建时间戳子目录。",
    )
    parser.add_argument(
        "--admin-token",
        default=os.getenv("ADMIN_API_TOKEN", ""),
        help="可选，访问 /api/admin/metrics 使用的 token。",
    )
    parser.add_argument(
        "--docker-container",
        action="append",
        default=[],
        help="可选，采集 docker stats 的容器名；可重复传入，也可逗号分隔。",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=1800.0,
        help="单次请求总超时，覆盖流式读完整轮。",
    )
    parser.add_argument(
        "--connect-timeout-seconds",
        type=float,
        default=15.0,
        help="连接超时。",
    )
    parser.add_argument(
        "--requested-model",
        default="",
        help="可选，强制透传到请求体的 model 字段。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="透传到请求体的 temperature。",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="透传到请求体的 top_p。",
    )
    parser.add_argument(
        "--run-label",
        default="ad-hoc",
        help="本次运行标签，例如 local-rehearsal 或 http-staging。",
    )
    parser.add_argument(
        "--skip-precheck",
        action="store_true",
        help="跳过预检，直接进入主压测与韧性场景。",
    )
    parser.add_argument(
        "--abort-on-precheck-fail",
        action="store_true",
        help="任一模式预检失败时立即中止整轮压测；默认仅记录告警并继续执行。",
    )
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def parse_concurrency(value: str) -> list[int]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("至少要提供一个并发档位。")
    values = [int(item) for item in items]
    if any(item <= 0 for item in values):
        raise ValueError("并发档位必须大于 0。")
    return values


def normalize_modes(value: str) -> list[Mode]:
    return ["rag"]


def normalize_docker_containers(values: list[str]) -> list[str]:
    merged: list[str] = []
    for item in values:
        merged.extend(part.strip() for part in item.split(",") if part.strip())
    return list(dict.fromkeys(merged))


def resolve_requests_per_stage(*, concurrency: int, fixed_requests: int, multiplier: float) -> int:
    if fixed_requests > 0:
        return fixed_requests
    if multiplier <= 0:
        raise ValueError("requests_per_stage_multiplier 必须大于 0。")
    return max(1, math.ceil(concurrency * multiplier))


def load_cases(path: Path) -> list[PerfCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("问题集文件必须是 JSON 数组。")
    rows: list[PerfCase] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"问题集第 {index + 1} 项不是对象。")
        case_id = str(item.get("id") or "").strip() or f"case_{index + 1:02d}"
        category = str(item.get("category") or "").strip()
        query = str(item.get("query") or "").strip()
        if not category or not query:
            raise ValueError(f"问题集第 {index + 1} 项缺少 category 或 query。")
        rows.append(PerfCase(case_id=case_id, category=category, query=query))
    if len(rows) < 12:
        raise ValueError("问题集至少需要 12 条，省得缓存和场景分布把结果测歪。")
    return rows


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def console_log(message: str) -> None:
    print(f"[perf] {message}", flush=True)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(math.ceil(len(ordered) * (q / 100.0)) - 1, 0)
    return round(ordered[index], 2)


def text_preview(value: str, limit: int = EVENT_PREVIEW_LIMIT) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def iso_from_epoch(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def safe_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [safe_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): safe_jsonable(item) for key, item in value.items()}
    return str(value)


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "avg": None,
            "stddev": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": len(values),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "avg": round(statistics.fmean(values), 2),
        "stddev": round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def summarize_errors(rows: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    samples: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw = str(row.get("error_text") or row.get("stream_error") or "").strip()
        if not raw:
            continue
        key = text_preview(raw, 200)
        counts[key] = counts.get(key, 0) + 1
        samples.setdefault(
            key,
            {
                "error_text": key,
                "http_status": row.get("http_status"),
                "trace_id": row.get("trace_id"),
                "case_id": row.get("case_id"),
                "category": row.get("category"),
            },
        )
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            **samples[key],
            "count": count,
        }
        for key, count in ordered[:limit]
    ]


def summarize_categories(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        category = str(row.get("category") or "unknown")
        buckets.setdefault(category, []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for category, bucket in buckets.items():
        total = len(bucket)
        success_count = sum(1 for row in bucket if row.get("success"))
        summary[category] = {
            "total": total,
            "success_count": success_count,
            "error_count": total - success_count,
            "error_rate": round((total - success_count) / total, 4) if total else 0.0,
            "status_counts": summarize_status(bucket, "done_status" if bucket[0].get("kind") == "stream" else "status"),
            "ttfb_ms": numeric_stats([float(row["ttfb_ms"]) for row in bucket if row.get("ttfb_ms") is not None]),
            "completed_ms": numeric_stats([float(row["completed_ms"]) for row in bucket if row.get("completed_ms") is not None]),
        }
    return summary


def summarize_event_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    message_events = [int(row.get("message_events") or 0) for row in rows]
    step_events = [int(row.get("step_events") or 0) for row in rows]
    chunk_count = [int(row.get("chunk_count") or 0) for row in rows]
    response_bytes = [int(row.get("response_bytes") or 0) for row in rows]
    response_chars = [int(row.get("response_text_chars") or 0) for row in rows]
    return {
        "message_events": numeric_stats([float(item) for item in message_events]),
        "step_events": numeric_stats([float(item) for item in step_events]),
        "chunk_count": numeric_stats([float(item) for item in chunk_count]),
        "response_bytes": numeric_stats([float(item) for item in response_bytes]),
        "response_text_chars": numeric_stats([float(item) for item in response_chars]),
    }


def build_stream_request(
    *,
    mode: Mode,
    case: PerfCase,
    requested_model: str,
    temperature: float,
    top_p: float,
    resilience: bool = False,
) -> dict[str, Any]:
    features: list[str]
    strict_citation = False
    request_mode = mode
    if resilience:
        features = ["rag", "skill_exec", "citation_guard"]
        strict_citation = True
        request_mode = "chat"
    elif mode == "rag":
        features = ["rag", "citation_guard"]
        strict_citation = True
    else:
        features = []
    payload: dict[str, Any] = {
        "session_id": uuid.uuid4().hex,
        "messages": [{"role": "user", "content": case.query}],
        "mode": request_mode,
        "stream": True,
        "features": features,
        "strict_citation": strict_citation,
        "temperature": temperature,
        "top_p": top_p,
    }
    if requested_model:
        payload["model"] = requested_model
    return payload


def build_sync_request(
    *,
    mode: Mode,
    case: PerfCase,
    requested_model: str,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    payload = build_stream_request(
        mode=mode,
        case=case,
        requested_model=requested_model,
        temperature=temperature,
        top_p=top_p,
    )
    payload["stream"] = False
    return payload


def parse_sse_block(block: str) -> tuple[str, str] | None:
    lines = [line.rstrip() for line in block.split("\n") if line.strip()]
    if not lines:
        return None
    event = "message"
    data_lines: list[str] = []
    for line in lines:
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    if not data_lines:
        return None
    return event, "\n".join(data_lines)


def extract_payload_error(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        message = detail.get("message")
        if isinstance(message, str):
            return message
    message = payload.get("message")
    if isinstance(message, str):
        return message
    error_message = payload.get("error_message")
    if isinstance(error_message, str):
        return error_message
    return ""


def detect_model_config_error(*, http_status: int, body_text: str, done_payload: dict[str, Any] | None) -> bool:
    if http_status == 400 and body_text.strip():
        return bool(MODEL_CONFIG_ERROR_PATTERN.search(body_text))
    if done_payload:
        error_message = str(done_payload.get("error_message") or "")
        if MODEL_CONFIG_ERROR_PATTERN.search(error_message):
            return True
    return False


def summarize_status(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "missing")
        summary[value] = summary.get(value, 0) + 1
    return summary


async def collect_http_snapshot(
    client: httpx.AsyncClient,
    *,
    admin_token: str,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"captured_at": utc_now()}
    health_error = ""
    try:
        response = await client.get("/healthz")
        snapshot["healthz_status_code"] = response.status_code
        snapshot["healthz"] = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
    except Exception as exc:  # noqa: BLE001
        health_error = str(exc)
        snapshot["healthz_error"] = health_error

    headers = {"x-admin-token": admin_token} if admin_token else {}
    try:
        response = await client.get("/api/admin/metrics", headers=headers)
        snapshot["admin_metrics_status_code"] = response.status_code
        snapshot["admin_metrics"] = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
    except Exception as exc:  # noqa: BLE001
        snapshot["admin_metrics_error"] = str(exc)
    return snapshot


def collect_docker_stats(container_names: list[str]) -> dict[str, Any]:
    if not container_names:
        return {"enabled": False, "containers": [], "error": ""}
    command = [
        "docker",
        "stats",
        "--no-stream",
        "--format",
        "{{json .}}",
        *container_names,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError:
        return {
            "enabled": True,
            "containers": [],
            "error": "docker 命令不存在，无法采集容器 CPU/内存/网络指标。",
        }
    if completed.returncode != 0:
        return {
            "enabled": True,
            "containers": [],
            "error": completed.stderr.strip() or "docker stats 执行失败。",
        }
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return {
        "enabled": True,
        "captured_at": utc_now(),
        "containers": rows,
        "error": "",
    }


async def collect_environment_snapshot(
    client: httpx.AsyncClient,
    *,
    admin_token: str,
    docker_containers: list[str],
) -> dict[str, Any]:
    http_snapshot = await collect_http_snapshot(client, admin_token=admin_token)
    return {
        "captured_at": utc_now(),
        "http": http_snapshot,
        "docker": collect_docker_stats(docker_containers),
    }


async def run_stream_request(
    client: httpx.AsyncClient,
    *,
    mode: Mode,
    case: PerfCase,
    requested_model: str,
    temperature: float,
    top_p: float,
    stage_name: str,
    concurrency: int,
    warmup: bool,
    ordinal: int,
    fail_features: str = "",
    resilience: bool = False,
) -> dict[str, Any]:
    payload = build_stream_request(
        mode=mode,
        case=case,
        requested_model=requested_model,
        temperature=temperature,
        top_p=top_p,
        resilience=resilience,
    )
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    if fail_features:
        headers["x-fail-features"] = fail_features

    started_epoch = time.time()
    started = time.perf_counter()
    http_status = 0
    response_text = ""
    done_payload: dict[str, Any] | None = None
    first_message_ms: float | None = None
    first_step_ms: float | None = None
    done_received_ms: float | None = None
    message_events = 0
    step_events = 0
    chunk_count = 0
    response_bytes = 0
    response_headers: dict[str, Any] = {}
    sse_events: list[dict[str, Any]] = []
    stream_error = ""
    missing_done = False
    try:
        async with client.stream("POST", "/api/chat/stream", json=payload, headers=headers) as response:
            http_status = response.status_code
            response_headers = {
                key.lower(): value
                for key, value in response.headers.items()
                if key.lower() in {"content-type", "content-length", "date", "server"}
            }
            if response.status_code != 200:
                raw = await response.aread()
                response_bytes = len(raw)
                response_text = raw.decode("utf-8", errors="replace")
            else:
                buffer = ""
                async for chunk in response.aiter_text():
                    if not chunk:
                        continue
                    chunk_count += 1
                    response_bytes += len(chunk.encode("utf-8"))
                    buffer += chunk.replace("\r\n", "\n")
                    while "\n\n" in buffer:
                        block, buffer = buffer.split("\n\n", 1)
                        parsed = parse_sse_block(block)
                        if not parsed:
                            continue
                        event, data = parsed
                        event_offset_ms = round((time.perf_counter() - started) * 1000, 2)
                        if event == "step":
                            step_events += 1
                            if first_step_ms is None:
                                first_step_ms = event_offset_ms
                            step_payload: dict[str, Any]
                            try:
                                step_payload = json.loads(data)
                            except json.JSONDecodeError:
                                step_payload = {"raw": data}
                            sse_events.append(
                                {
                                    "seq": len(sse_events) + 1,
                                    "event": "step",
                                    "offset_ms": event_offset_ms,
                                    "payload_size_chars": len(data),
                                    "node": step_payload.get("node"),
                                    "title": step_payload.get("title"),
                                    "status": step_payload.get("status"),
                                    "attempt": step_payload.get("attempt"),
                                    "message_preview": text_preview(str(step_payload.get("message") or "")),
                                }
                            )
                            continue
                        if event == "done":
                            try:
                                done_payload = json.loads(data)
                            except json.JSONDecodeError:
                                done_payload = {"raw": data}
                            done_received_ms = event_offset_ms
                            sse_events.append(
                                {
                                    "seq": len(sse_events) + 1,
                                    "event": "done",
                                    "offset_ms": event_offset_ms,
                                    "payload_size_chars": len(data),
                                    "status": done_payload.get("status"),
                                    "trace_id": done_payload.get("trace_id"),
                                    "finish_reason": done_payload.get("finish_reason"),
                                    "degraded_features": done_payload.get("degraded_features", []),
                                }
                            )
                            break
                        message_events += 1
                        if first_message_ms is None:
                            first_message_ms = event_offset_ms
                        message_payload: dict[str, Any]
                        try:
                            message_payload = json.loads(data)
                        except json.JSONDecodeError:
                            message_payload = {"raw": data}
                        delta = str(message_payload.get("delta") or data)
                        sse_events.append(
                            {
                                "seq": len(sse_events) + 1,
                                "event": "message",
                                "offset_ms": event_offset_ms,
                                "payload_size_chars": len(data),
                                "delta_chars": len(delta),
                                "delta_preview": text_preview(delta),
                            }
                        )
                    if done_payload is not None:
                        break
                if done_payload is None and buffer.strip():
                    parsed = parse_sse_block(buffer.strip())
                    if parsed and parsed[0] == "done":
                        try:
                            done_payload = json.loads(parsed[1])
                        except json.JSONDecodeError:
                            done_payload = {"raw": parsed[1]}
                        done_received_ms = round((time.perf_counter() - started) * 1000, 2)
                        sse_events.append(
                            {
                                "seq": len(sse_events) + 1,
                                "event": "done",
                                "offset_ms": done_received_ms,
                                "payload_size_chars": len(parsed[1]),
                                "status": done_payload.get("status"),
                                "trace_id": done_payload.get("trace_id"),
                                "finish_reason": done_payload.get("finish_reason"),
                                "degraded_features": done_payload.get("degraded_features", []),
                            }
                        )
                missing_done = done_payload is None
    except Exception as exc:  # noqa: BLE001
        stream_error = str(exc)

    finished_epoch = time.time()
    completed_ms = round((time.perf_counter() - started) * 1000, 2)
    done_status = str(done_payload.get("status") or "") if done_payload else ""
    done_trace_id = str(done_payload.get("trace_id") or "") if done_payload else ""
    done_degraded = done_payload.get("degraded_features", []) if done_payload else []
    error_text = stream_error or response_text
    if done_payload and not error_text:
        error_text = str(done_payload.get("error_message") or "")
    model_config_error = detect_model_config_error(
        http_status=http_status,
        body_text=error_text,
        done_payload=done_payload,
    )
    succeeded = (
        not resilience
        and http_status == 200
        and not missing_done
        and done_status == "ok"
        and not model_config_error
        and not stream_error
    )
    degraded_expected = resilience and http_status == 200 and not missing_done and done_status == "degraded"
    return {
        "kind": "stream",
        "mode": mode,
        "stage_name": stage_name,
        "category": case.category,
        "case_id": case.case_id,
        "query": case.query,
        "request_started_at": iso_from_epoch(started_epoch),
        "request_completed_at": iso_from_epoch(finished_epoch),
        "request_started_epoch": round(started_epoch, 6),
        "request_completed_epoch": round(finished_epoch, 6),
        "ordinal": ordinal,
        "concurrency": concurrency,
        "warmup": warmup,
        "resilience": resilience,
        "fail_features": fail_features,
        "session_id": payload["session_id"],
        "request_mode": payload["mode"],
        "request_stream": payload["stream"],
        "request_features": payload["features"],
        "strict_citation": payload["strict_citation"],
        "requested_model": payload.get("model", ""),
        "temperature": payload["temperature"],
        "top_p": payload["top_p"],
        "request_headers": headers,
        "http_status": http_status,
        "response_headers": response_headers,
        "first_step_ms": first_step_ms,
        "ttfb_ms": first_message_ms,
        "done_received_ms": done_received_ms,
        "completed_ms": completed_ms,
        "message_events": message_events,
        "step_events": step_events,
        "chunk_count": chunk_count,
        "response_bytes": response_bytes,
        "response_text_chars": len(response_text),
        "done_status": done_status,
        "degraded_features": done_degraded,
        "trace_id": done_trace_id,
        "missing_done": missing_done,
        "model_config_error": model_config_error,
        "error_text": error_text,
        "response_text_preview": text_preview(response_text, 240),
        "stream_error": stream_error,
        "sse_events": sse_events,
        "done_payload": safe_jsonable(done_payload) if done_payload else None,
        "success": succeeded,
        "resilience_success": degraded_expected,
    }


async def run_sync_request(
    client: httpx.AsyncClient,
    *,
    mode: Mode,
    case: PerfCase,
    requested_model: str,
    temperature: float,
    top_p: float,
    ordinal: int,
) -> dict[str, Any]:
    payload = build_sync_request(
        mode=mode,
        case=case,
        requested_model=requested_model,
        temperature=temperature,
        top_p=top_p,
    )
    headers = {"Content-Type": "application/json"}
    started_epoch = time.time()
    started = time.perf_counter()
    http_status = 0
    body: dict[str, Any] | None = None
    response_text = ""
    request_error = ""
    response_headers: dict[str, Any] = {}
    response_bytes = 0
    try:
        response = await client.post("/api/chat", json=payload, headers=headers)
        http_status = response.status_code
        response_headers = {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower() in {"content-type", "content-length", "date", "server"}
        }
        response_bytes = len(response.content)
        if response.headers.get("content-type", "").startswith("application/json"):
            body = response.json()
        else:
            response_text = response.text
    except Exception as exc:  # noqa: BLE001
        request_error = str(exc)
    finished_epoch = time.time()
    completed_ms = round((time.perf_counter() - started) * 1000, 2)
    body_status = str(body.get("status") or "") if body else ""
    if not response_text and body is not None:
        response_text = json.dumps(body, ensure_ascii=False)
    model_config_error = detect_model_config_error(
        http_status=http_status,
        body_text=request_error or response_text or extract_payload_error(body),
        done_payload=None,
    )
    success = http_status == 200 and body_status == "ok" and not model_config_error and not request_error
    return {
        "kind": "sync",
        "mode": mode,
        "stage_name": f"precheck_sync_{mode}",
        "category": case.category,
        "case_id": case.case_id,
        "query": case.query,
        "request_started_at": iso_from_epoch(started_epoch),
        "request_completed_at": iso_from_epoch(finished_epoch),
        "request_started_epoch": round(started_epoch, 6),
        "request_completed_epoch": round(finished_epoch, 6),
        "ordinal": ordinal,
        "session_id": payload["session_id"],
        "request_mode": payload["mode"],
        "request_stream": payload["stream"],
        "request_features": payload["features"],
        "strict_citation": payload["strict_citation"],
        "requested_model": payload.get("model", ""),
        "temperature": payload["temperature"],
        "top_p": payload["top_p"],
        "request_headers": headers,
        "http_status": http_status,
        "response_headers": response_headers,
        "response_bytes": response_bytes,
        "response_text_chars": len(response_text),
        "completed_ms": completed_ms,
        "status": body_status,
        "trace_id": str(body.get("trace_id") or "") if body else "",
        "degraded_features": body.get("degraded_features", []) if body else [],
        "model_config_error": model_config_error,
        "error_text": request_error or extract_payload_error(body) or response_text,
        "response_text_preview": text_preview(response_text, 240),
        "response_body": safe_jsonable(body) if body else None,
        "success": success,
    }


async def run_stream_batch(
    client: httpx.AsyncClient,
    *,
    mode: Mode,
    cases: list[PerfCase],
    requested_model: str,
    temperature: float,
    top_p: float,
    stage_name: str,
    concurrency: int,
    total_requests: int,
    case_offset: int,
    warmup: bool,
    fail_features: str = "",
    resilience: bool = False,
    progress_label: str = "",
) -> tuple[list[dict[str, Any]], int]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any] | None] = [None] * total_requests
    stats = {
        "ok": 0,
        "fail": 0,
        "degraded": 0,
        "missing_done": 0,
    }
    progress = tqdm(
        total=total_requests,
        desc=progress_label or stage_name,
        unit="req",
        leave=True,
        dynamic_ncols=True,
    )

    async def worker(index: int) -> None:
        async with semaphore:
            case = cases[(case_offset + index) % len(cases)]
            row = await run_stream_request(
                client,
                mode=mode,
                case=case,
                requested_model=requested_model,
                temperature=temperature,
                top_p=top_p,
                stage_name=stage_name,
                concurrency=concurrency,
                warmup=warmup,
                ordinal=case_offset + index,
                fail_features=fail_features,
                resilience=resilience,
            )
            results[index] = row
            if row.get("success") or row.get("resilience_success"):
                stats["ok"] += 1
            else:
                stats["fail"] += 1
            if row.get("done_status") == "degraded":
                stats["degraded"] += 1
            if row.get("missing_done"):
                stats["missing_done"] += 1
            progress.update(1)
            progress.set_postfix(
                ok=stats["ok"],
                fail=stats["fail"],
                degraded=stats["degraded"],
                miss_done=stats["missing_done"],
                refresh=False,
            )

    try:
        await asyncio.gather(*(worker(index) for index in range(total_requests)))
    finally:
        progress.close()
    return [item for item in results if item is not None], case_offset + total_requests


def summarize_precheck(stream_rows: list[dict[str, Any]], sync_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stream_passed = all(row["success"] for row in stream_rows)
    sync_passed = all(row["success"] for row in sync_rows)
    stream_model_errors = sum(1 for row in stream_rows if row["model_config_error"])
    sync_model_errors = sum(1 for row in sync_rows if row["model_config_error"])
    missing_done_count = sum(1 for row in stream_rows if row["missing_done"])
    sync_p95 = percentile([float(row["completed_ms"]) for row in sync_rows], 95)
    sync_gate_passed = sync_p95 is not None and sync_p95 <= 1500
    return {
        "stream_requests": len(stream_rows),
        "sync_requests": len(sync_rows),
        "stream_passed": stream_passed,
        "sync_passed": sync_passed,
        "missing_done_count": missing_done_count,
        "stream_model_config_errors": stream_model_errors,
        "sync_model_config_errors": sync_model_errors,
        "sync_p95_ms": sync_p95,
        "sync_gate_passed": sync_gate_passed,
        "stream_ttfb_ms": numeric_stats([float(row["ttfb_ms"]) for row in stream_rows if row.get("ttfb_ms") is not None]),
        "stream_completed_ms": numeric_stats([float(row["completed_ms"]) for row in stream_rows if row.get("completed_ms") is not None]),
        "sync_completed_ms": numeric_stats([float(row["completed_ms"]) for row in sync_rows if row.get("completed_ms") is not None]),
        "stream_event_stats": summarize_event_counts(stream_rows),
        "stream_category_breakdown": summarize_categories(stream_rows),
        "sync_category_breakdown": summarize_categories(sync_rows),
        "error_summary": {
            "stream": summarize_errors(stream_rows),
            "sync": summarize_errors(sync_rows),
        },
        "passed": (
            stream_passed
            and sync_passed
            and missing_done_count == 0
            and stream_model_errors == 0
            and sync_model_errors == 0
            and sync_gate_passed
        ),
        "stream_status_counts": summarize_status(stream_rows, "done_status"),
        "sync_status_counts": summarize_status(sync_rows, "status"),
    }


def summarize_main_stage(rows: list[dict[str, Any]], *, mode: Mode, concurrency: int) -> dict[str, Any]:
    measured = [row for row in rows if not row["warmup"]]
    warmup_rows = [row for row in rows if row["warmup"]]
    total = len(measured)
    success_count = sum(1 for row in measured if row["success"])
    degraded_count = sum(1 for row in measured if row.get("done_status") == "degraded")
    failed_count = sum(1 for row in measured if row.get("done_status") == "failed")
    missing_done_count = sum(1 for row in measured if row["missing_done"])
    model_config_errors = sum(1 for row in measured if row["model_config_error"])
    error_rate = round((total - success_count) / total, 4) if total else 0.0
    ttfb_values = [float(row["ttfb_ms"]) for row in measured if row["ttfb_ms"] is not None]
    ttfb_stats = numeric_stats(ttfb_values)
    p95_ttfb_ms = percentile(ttfb_values, 95)
    p95_completed_ms = percentile([float(row["completed_ms"]) for row in measured], 95)

    threshold_error_rate = 0.0 if concurrency <= 15 else 0.01
    passed = missing_done_count == 0 and error_rate <= threshold_error_rate and model_config_errors == 0
    acceptance_checks = [
        {
            "name": "error_rate",
            "actual": error_rate,
            "expected": f"<={threshold_error_rate}",
            "passed": error_rate <= threshold_error_rate,
        },
        {
            "name": "missing_done_count",
            "actual": missing_done_count,
            "expected": "==0",
            "passed": missing_done_count == 0,
        },
        {
            "name": "model_config_errors",
            "actual": model_config_errors,
            "expected": "==0",
            "passed": model_config_errors == 0,
        },
    ]
    if mode == "rag":
        ttfb_ok = p95_ttfb_ms is not None and p95_ttfb_ms <= 3000
        completed_ok = p95_completed_ms is not None and p95_completed_ms <= 12000
        passed = passed and ttfb_ok and completed_ok
        acceptance_checks.extend(
            [
                {
                    "name": "p95_ttfb_ms",
                    "actual": p95_ttfb_ms,
                    "expected": "<=3000",
                    "passed": ttfb_ok,
                },
                {
                    "name": "p95_completed_ms",
                    "actual": p95_completed_ms,
                    "expected": "<=12000",
                    "passed": completed_ok,
                },
            ]
        )
    return {
        "mode": mode,
        "concurrency": concurrency,
        "warmup_requests": len(warmup_rows),
        "total_requests": total,
        "success_count": success_count,
        "degraded_count": degraded_count,
        "failed_count": failed_count,
        "error_rate": error_rate,
        "missing_done_count": missing_done_count,
        "model_config_errors": model_config_errors,
        "p95_ttfb_ms": p95_ttfb_ms,
        "p95_completed_ms": p95_completed_ms,
        "ttfb_ms": ttfb_stats,
        "ttfb_avg_ms": ttfb_stats.get("avg"),
        "ttfb_max_ms": ttfb_stats.get("max"),
        "completed_ms": numeric_stats([float(row["completed_ms"]) for row in measured if row.get("completed_ms") is not None]),
        "first_step_ms": numeric_stats([float(row["first_step_ms"]) for row in measured if row.get("first_step_ms") is not None]),
        "done_received_ms": numeric_stats([float(row["done_received_ms"]) for row in measured if row.get("done_received_ms") is not None]),
        "event_stats": summarize_event_counts(measured),
        "done_status_counts": summarize_status(measured, "done_status"),
        "http_status_counts": summarize_status(measured, "http_status"),
        "category_breakdown": summarize_categories(measured),
        "error_summary": summarize_errors(measured),
        "passed": passed,
        "acceptance_checks": acceptance_checks,
    }


def summarize_resilience_stage(rows: list[dict[str, Any]], *, fail_features: str) -> dict[str, Any]:
    total = len(rows)
    degraded_count = sum(1 for row in rows if row["done_status"] == "degraded")
    http_5xx_count = sum(1 for row in rows if int(row["http_status"]) >= 500)
    missing_done_count = sum(1 for row in rows if row["missing_done"])
    passed = total > 0 and degraded_count == total and http_5xx_count == 0 and missing_done_count == 0
    return {
        "fail_features": fail_features,
        "total_requests": total,
        "degraded_count": degraded_count,
        "http_5xx_count": http_5xx_count,
        "missing_done_count": missing_done_count,
        "done_status_counts": summarize_status(rows, "done_status"),
        "completed_ms": numeric_stats([float(row["completed_ms"]) for row in rows if row.get("completed_ms") is not None]),
        "ttfb_ms": numeric_stats([float(row["ttfb_ms"]) for row in rows if row.get("ttfb_ms") is not None]),
        "event_stats": summarize_event_counts(rows),
        "category_breakdown": summarize_categories(rows),
        "error_summary": summarize_errors(rows),
        "passed": passed,
    }


def build_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# 并发压测摘要",
        "",
        f"- 运行标签：`{summary['run_label']}`",
        f"- 目标地址：`{summary['base_url']}`",
        f"- 运行时间：`{summary['started_at']}` -> `{summary['finished_at']}`",
        f"- 模式：`{', '.join(summary['modes'])}`",
        f"- 总体结果：`{'PASS' if summary['passed'] else 'FAIL'}`",
        "",
        "## 预检",
    ]
    for mode in summary["modes"]:
        precheck_entry = summary.get("precheck", {}).get(mode)
        if not precheck_entry:
            lines.extend(
                [
                    f"### {mode}",
                    "- 本模式未执行预检，或在预检前提前结束。",
                    "",
                ]
            )
            continue
        precheck = precheck_entry["summary"]
        lines.extend(
            [
                f"### {mode}",
                f"- 流式预检：{'通过' if precheck['stream_passed'] else '失败'}，次数 `{precheck['stream_requests']}`，缺失 done `{precheck['missing_done_count']}`",
                f"- 同步预检：{'通过' if precheck['sync_passed'] else '失败'}，次数 `{precheck['sync_requests']}`，P95 `{precheck['sync_p95_ms']}` ms，门禁 `{'通过' if precheck.get('sync_gate_passed', False) else '失败'}`",
                f"- 模型配置异常：流式 `{precheck['stream_model_config_errors']}`，同步 `{precheck['sync_model_config_errors']}`",
                f"- 流式耗时：P50 `{precheck.get('stream_completed_ms', {}).get('p50')}` ms，P95 `{precheck.get('stream_completed_ms', {}).get('p95')}` ms；事件均值 `{precheck.get('stream_event_stats', {}).get('message_events', {}).get('avg')}` 条",
                "",
            ]
        )

    lines.append("## 主压测")
    for mode in summary["modes"]:
        lines.append(f"### {mode}")
        for stage in summary.get("main", {}).get(mode, []):
            stage_summary = stage["summary"]
            lines.append(
                f"- 并发 `{stage_summary['concurrency']}`：`{'PASS' if stage_summary['passed'] else 'FAIL'}`，"
                f"错误率 `{stage_summary['error_rate']}`，P95 TTFB `{stage_summary['p95_ttfb_ms']}` ms，"
                f"P95 完整耗时 `{stage_summary['p95_completed_ms']}` ms，P99 `{stage_summary.get('completed_ms', {}).get('p99')}` ms，"
                f"缺失 done `{stage_summary['missing_done_count']}`，degraded `{stage_summary.get('degraded_count')}`"
            )
        if not summary.get("main", {}).get(mode):
            lines.append("- 本模式未进入主压测。")
        lines.append("")

    lines.append("## 韧性场景")
    for item in summary["resilience"]:
        lines.append(
            f"- 故障注入 `{item['summary']['fail_features']}`：`{'PASS' if item['summary']['passed'] else 'FAIL'}`，"
            f"degraded `{item['summary']['degraded_count']}/{item['summary']['total_requests']}`，"
            f"HTTP 5xx `{item['summary']['http_5xx_count']}`"
        )
    lines.append("")

    if summary["warnings"]:
        lines.append("## 备注")
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


async def run_mode_precheck(
    client: httpx.AsyncClient,
    *,
    mode: Mode,
    cases: list[PerfCase],
    requested_model: str,
    temperature: float,
    top_p: float,
    stream_requests: int,
    sync_requests: int,
    admin_token: str,
    docker_containers: list[str],
    case_offset: int,
) -> tuple[dict[str, Any], int]:
    console_log(
        f"开始预检 mode={mode} stream_requests={stream_requests} sync_requests={sync_requests}"
    )
    before_snapshot = await collect_environment_snapshot(
        client,
        admin_token=admin_token,
        docker_containers=docker_containers,
    )
    stream_rows: list[dict[str, Any]] = []
    stream_progress = tqdm(
        total=stream_requests,
        desc=f"precheck-stream-{mode}",
        unit="req",
        leave=True,
        dynamic_ncols=True,
    )
    try:
        for index in range(stream_requests):
            case = cases[(case_offset + index) % len(cases)]
            row = await run_stream_request(
                client,
                mode=mode,
                case=case,
                requested_model=requested_model,
                temperature=temperature,
                top_p=top_p,
                stage_name=f"precheck_stream_{mode}",
                concurrency=1,
                warmup=False,
                ordinal=case_offset + index,
            )
            stream_rows.append(row)
            stream_progress.update(1)
            stream_progress.set_postfix(
                ok=sum(1 for item in stream_rows if item["success"]),
                fail=sum(1 for item in stream_rows if not item["success"]),
                miss_done=sum(1 for item in stream_rows if item["missing_done"]),
                refresh=False,
            )
    finally:
        stream_progress.close()
    case_offset += stream_requests

    sync_rows: list[dict[str, Any]] = []
    sync_progress = tqdm(
        total=sync_requests,
        desc=f"precheck-sync-{mode}",
        unit="req",
        leave=True,
        dynamic_ncols=True,
    )
    try:
        for index in range(sync_requests):
            case = cases[(case_offset + index) % len(cases)]
            row = await run_sync_request(
                client,
                mode=mode,
                case=case,
                requested_model=requested_model,
                temperature=temperature,
                top_p=top_p,
                ordinal=case_offset + index,
            )
            sync_rows.append(row)
            sync_progress.update(1)
            sync_progress.set_postfix(
                ok=sum(1 for item in sync_rows if item["success"]),
                fail=sum(1 for item in sync_rows if not item["success"]),
                refresh=False,
            )
    finally:
        sync_progress.close()
    case_offset += sync_requests
    after_snapshot = await collect_environment_snapshot(
        client,
        admin_token=admin_token,
        docker_containers=docker_containers,
    )

    precheck_summary = summarize_precheck(stream_rows, sync_rows)
    console_log(
        "预检完成 "
        f"mode={mode} passed={precheck_summary['passed']} "
        f"stream_ok={precheck_summary['stream_passed']} "
        f"sync_ok={precheck_summary['sync_passed']} "
        f"sync_p95_ms={precheck_summary['sync_p95_ms']}"
    )

    return {
        "summary": precheck_summary,
        "stream_rows": stream_rows,
        "sync_rows": sync_rows,
        "environment": {
            "environment_before": before_snapshot,
            "environment_after": after_snapshot,
        },
    }, case_offset


async def run_mode_stages(
    client: httpx.AsyncClient,
    *,
    mode: Mode,
    cases: list[PerfCase],
    requested_model: str,
    temperature: float,
    top_p: float,
    concurrency_values: list[int],
    warmup_requests: int,
    requests_per_stage: int,
    requests_per_stage_multiplier: float,
    cooldown_seconds: float,
    admin_token: str,
    docker_containers: list[str],
    case_offset: int,
) -> tuple[list[dict[str, Any]], int]:
    stages: list[dict[str, Any]] = []
    for index, concurrency in enumerate(concurrency_values):
        stage_name = f"{mode}_c{concurrency}"
        stage_requests_per_stage = resolve_requests_per_stage(
            concurrency=concurrency,
            fixed_requests=requests_per_stage,
            multiplier=requests_per_stage_multiplier,
        )
        console_log(
            f"开始主压测 stage={stage_name} warmup={warmup_requests} measured={stage_requests_per_stage}"
        )
        before_snapshot = await collect_environment_snapshot(
            client,
            admin_token=admin_token,
            docker_containers=docker_containers,
        )
        warmup_rows, case_offset = await run_stream_batch(
            client,
            mode=mode,
            cases=cases,
            requested_model=requested_model,
            temperature=temperature,
            top_p=top_p,
            stage_name=stage_name,
            concurrency=concurrency,
            total_requests=warmup_requests,
            case_offset=case_offset,
            warmup=True,
            progress_label=f"{stage_name}-warmup",
        )
        measured_rows, case_offset = await run_stream_batch(
            client,
            mode=mode,
            cases=cases,
            requested_model=requested_model,
            temperature=temperature,
            top_p=top_p,
            stage_name=stage_name,
            concurrency=concurrency,
            total_requests=stage_requests_per_stage,
            case_offset=case_offset,
            warmup=False,
            progress_label=f"{stage_name}-measured",
        )
        after_snapshot = await collect_environment_snapshot(
            client,
            admin_token=admin_token,
            docker_containers=docker_containers,
        )
        rows = [*warmup_rows, *measured_rows]
        stage_summary = summarize_main_stage(rows, mode=mode, concurrency=concurrency)
        console_log(
            f"阶段完成 stage={stage_name} passed={stage_summary['passed']} "
            f"error_rate={stage_summary['error_rate']} "
            f"p95_ttfb_ms={stage_summary['p95_ttfb_ms']} "
            f"p95_completed_ms={stage_summary['p95_completed_ms']}"
        )
        stages.append(
            {
                "stage_name": stage_name,
                "summary": stage_summary,
                "environment": {
                    "before": before_snapshot,
                    "after": after_snapshot,
                },
                "rows": rows,
            }
        )
        if index != len(concurrency_values) - 1 and cooldown_seconds > 0:
            console_log(f"阶段冷却 {cooldown_seconds} 秒 stage={stage_name}")
            await asyncio.sleep(cooldown_seconds)
    return stages, case_offset


async def run_resilience_checks(
    client: httpx.AsyncClient,
    *,
    mode: Mode,
    cases: list[PerfCase],
    requested_model: str,
    temperature: float,
    top_p: float,
    admin_token: str,
    docker_containers: list[str],
    case_offset: int,
) -> tuple[list[dict[str, Any]], int]:
    scenarios: list[dict[str, Any]] = []
    for fail_features in ("rag", "skill_exec"):
        console_log(f"开始韧性场景 fail_features={fail_features} concurrency=8")
        before_snapshot = await collect_environment_snapshot(
            client,
            admin_token=admin_token,
            docker_containers=docker_containers,
        )
        rows, case_offset = await run_stream_batch(
            client,
            mode=mode,
            cases=cases,
            requested_model=requested_model,
            temperature=temperature,
            top_p=top_p,
            stage_name=f"resilience_{fail_features}",
            concurrency=8,
            total_requests=8,
            case_offset=case_offset,
            warmup=False,
            fail_features=fail_features,
            resilience=True,
            progress_label=f"resilience-{fail_features}",
        )
        after_snapshot = await collect_environment_snapshot(
            client,
            admin_token=admin_token,
            docker_containers=docker_containers,
        )
        resilience_summary = summarize_resilience_stage(rows, fail_features=fail_features)
        console_log(
            f"韧性场景完成 fail_features={fail_features} "
            f"passed={resilience_summary['passed']} "
            f"degraded={resilience_summary['degraded_count']}/{resilience_summary['total_requests']} "
            f"http_5xx={resilience_summary['http_5xx_count']}"
        )
        scenarios.append(
            {
                "summary": resilience_summary,
                "environment": {
                    "before": before_snapshot,
                    "after": after_snapshot,
                },
                "rows": rows,
            }
        )
    return scenarios, case_offset


def flatten_request_rows(raw_rows: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, payload in raw_rows.get("precheck", {}).items():
        for item in payload.get("stream_rows", []):
            rows.append({"phase": "precheck", **item})
        for item in payload.get("sync_rows", []):
            rows.append({"phase": "precheck", **item})
    for mode, stages in raw_rows.get("main", {}).items():
        for stage in stages:
            for item in stage.get("rows", []):
                rows.append({"phase": "main", **item})
    for scenario in raw_rows.get("resilience", []):
        for item in scenario.get("rows", []):
            rows.append({"phase": "resilience", **item})
    return rows


def flatten_sse_events(request_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in request_rows:
        for event in row.get("sse_events", []):
            events.append(
                {
                    "phase": row.get("phase"),
                    "stage_name": row.get("stage_name"),
                    "mode": row.get("mode"),
                    "concurrency": row.get("concurrency"),
                    "warmup": row.get("warmup"),
                    "resilience": row.get("resilience"),
                    "fail_features": row.get("fail_features"),
                    "session_id": row.get("session_id"),
                    "trace_id": row.get("trace_id"),
                    "case_id": row.get("case_id"),
                    "category": row.get("category"),
                    **event,
                }
            )
    return events


def flatten_stage_summaries(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, payload in summary.get("precheck", {}).items():
        precheck = payload.get("summary", {})
        rows.append(
            {
                "phase": "precheck",
                "mode": mode,
                "stage_name": f"precheck_{mode}",
                "concurrency": 1,
                "passed": precheck.get("passed"),
                "stream_requests": precheck.get("stream_requests"),
                "sync_requests": precheck.get("sync_requests"),
                "sync_p95_ms": precheck.get("sync_p95_ms"),
                "missing_done_count": precheck.get("missing_done_count"),
                "error_rate": 0.0 if precheck.get("passed") else None,
                "p95_ttfb_ms": precheck.get("stream_ttfb_ms", {}).get("p95"),
                "ttfb_avg_ms": precheck.get("stream_ttfb_ms", {}).get("avg"),
                "ttfb_max_ms": precheck.get("stream_ttfb_ms", {}).get("max"),
                "p95_completed_ms": precheck.get("stream_completed_ms", {}).get("p95"),
            }
        )
    for mode, stages in summary.get("main", {}).items():
        for stage in stages:
            row = stage.get("summary", {})
            rows.append(
                {
                    "phase": "main",
                    "mode": mode,
                    "stage_name": stage.get("stage_name"),
                    "concurrency": row.get("concurrency"),
                    "passed": row.get("passed"),
                    "stream_requests": row.get("total_requests"),
                    "sync_requests": 0,
                    "sync_p95_ms": None,
                    "missing_done_count": row.get("missing_done_count"),
                    "error_rate": row.get("error_rate"),
                    "p95_ttfb_ms": row.get("p95_ttfb_ms"),
                    "ttfb_avg_ms": row.get("ttfb_avg_ms"),
                    "ttfb_max_ms": row.get("ttfb_max_ms"),
                    "p95_completed_ms": row.get("p95_completed_ms"),
                }
            )
    for item in summary.get("resilience", []):
        row = item.get("summary", {})
        rows.append(
            {
                "phase": "resilience",
                "mode": "rag",
                "stage_name": f"resilience_{row.get('fail_features')}",
                "concurrency": 8,
                "passed": row.get("passed"),
                "stream_requests": row.get("total_requests"),
                "sync_requests": 0,
                "sync_p95_ms": None,
                "missing_done_count": row.get("missing_done_count"),
                "error_rate": None,
                "p95_ttfb_ms": row.get("ttfb_ms", {}).get("p95"),
                "ttfb_avg_ms": row.get("ttfb_ms", {}).get("avg"),
                "ttfb_max_ms": row.get("ttfb_ms", {}).get("max"),
                "p95_completed_ms": row.get("completed_ms", {}).get("p95"),
            }
        )
    return rows


def flatten_environment_snapshots(raw_rows: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, payload in raw_rows.get("precheck", {}).items():
        env = payload.get("environment")
        if env:
            rows.append({"phase": "precheck", "mode": mode, "stage_name": f"precheck_{mode}", **env})
    for mode, stages in raw_rows.get("main", {}).items():
        for stage in stages:
            environment = stage.get("environment", {})
            rows.append(
                {
                    "phase": "main",
                    "mode": mode,
                    "stage_name": stage.get("stage_name"),
                    "environment_before": environment.get("before"),
                    "environment_after": environment.get("after"),
                }
            )
    for scenario in raw_rows.get("resilience", []):
        environment = scenario.get("environment", {})
        rows.append(
            {
                "phase": "resilience",
                "mode": "rag",
                "stage_name": f"resilience_{scenario.get('summary', {}).get('fail_features')}",
                "environment_before": environment.get("before"),
                "environment_after": environment.get("after"),
            }
        )
    run_environment = raw_rows.get("run_environment")
    if run_environment:
        rows.append({"phase": "run", "mode": "all", "stage_name": "run_scope", **run_environment})
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(safe_jsonable(row), ensure_ascii=False) + "\n")


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flattened = {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(flattened)


def persist_reports(output_dir: Path, *, summary: dict[str, Any], raw_rows: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    request_rows = flatten_request_rows(raw_rows)
    sse_events = flatten_sse_events(request_rows)
    stage_rows = flatten_stage_summaries(summary)
    environment_rows = flatten_environment_snapshots(raw_rows)
    (output_dir / "perf_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "perf_raw.json").write_text(json.dumps(raw_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "perf_summary.md").write_text(build_markdown_summary(summary), encoding="utf-8")
    write_jsonl(output_dir / "request_rows.jsonl", request_rows)
    write_jsonl(output_dir / "sse_events.jsonl", sse_events)
    write_jsonl(output_dir / "environment_snapshots.jsonl", environment_rows)
    write_csv_rows(output_dir / "request_rows.csv", request_rows)
    write_csv_rows(output_dir / "stage_summaries.csv", stage_rows)


async def async_main(args: argparse.Namespace) -> int:
    cases_file = resolve_path(args.cases_file)
    output_root = resolve_path(args.output)
    base_url = args.base_url.rstrip("/")
    modes = normalize_modes(args.mode)
    concurrency_values = parse_concurrency(args.concurrency)
    docker_containers = normalize_docker_containers(args.docker_container)
    cases = load_cases(cases_file)
    max_concurrency = max(max(concurrency_values), 8)

    run_dir = output_root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.run_label}"
    timeout = httpx.Timeout(
        timeout=args.request_timeout_seconds,
        connect=args.connect_timeout_seconds,
        read=args.request_timeout_seconds,
    )
    limits = httpx.Limits(
        max_connections=max(max_concurrency * 2, 20),
        max_keepalive_connections=max(max_concurrency, 10),
    )

    started_at = utc_now()
    raw_rows: dict[str, Any] = {
        "config": {
            "run_label": args.run_label,
            "base_url": base_url,
            "modes": modes,
            "concurrency_values": concurrency_values,
            "warmup_requests": args.warmup_requests,
            "requests_per_stage": args.requests_per_stage,
            "requests_per_stage_multiplier": args.requests_per_stage_multiplier,
            "cooldown_seconds": args.cooldown_seconds,
            "precheck_stream_requests": args.precheck_stream_requests,
            "precheck_sync_requests": args.precheck_sync_requests,
            "request_timeout_seconds": args.request_timeout_seconds,
            "connect_timeout_seconds": args.connect_timeout_seconds,
            "requested_model": args.requested_model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "cases_file": str(cases_file),
            "output_dir": str(run_dir),
            "docker_containers": docker_containers,
        },
        "cases": [case.__dict__ for case in cases],
        "run_environment": {},
        "precheck": {},
        "main": {},
        "resilience": [],
    }
    summary: dict[str, Any] = {
        "run_label": args.run_label,
        "base_url": base_url,
        "started_at": started_at,
        "finished_at": "",
        "modes": modes,
        "concurrency_values": concurrency_values,
        "warmup_requests": args.warmup_requests,
        "requests_per_stage": args.requests_per_stage,
        "requests_per_stage_multiplier": args.requests_per_stage_multiplier,
        "precheck": {},
        "main": {},
        "resilience": [],
        "warnings": [],
        "passed": False,
        "skip_precheck": args.skip_precheck,
    }

    console_log(
        f"启动压测 run_label={args.run_label} base_url={base_url} modes={modes} "
        f"concurrency={concurrency_values} cases={len(cases)} output={run_dir} "
        f"skip_precheck={args.skip_precheck}"
    )
    case_offset = 0
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, limits=limits, follow_redirects=True) as client:
        raw_rows["run_environment"]["environment_before"] = await collect_environment_snapshot(
            client,
            admin_token=args.admin_token,
            docker_containers=docker_containers,
        )
        for mode in modes:
            if args.skip_precheck:
                console_log(f"跳过预检 mode={mode}")
                summary["precheck"][mode] = {
                    "summary": {
                        "passed": True,
                        "skipped": True,
                        "stream_passed": True,
                        "sync_passed": True,
                        "stream_requests": 0,
                        "sync_requests": 0,
                        "missing_done_count": 0,
                        "stream_model_config_errors": 0,
                        "sync_model_config_errors": 0,
                        "sync_p95_ms": None,
                        "sync_gate_passed": True,
                        "stream_completed_ms": numeric_stats([]),
                        "stream_ttfb_ms": numeric_stats([]),
                        "sync_completed_ms": numeric_stats([]),
                        "stream_event_stats": summarize_event_counts([]),
                        "stream_category_breakdown": {},
                        "sync_category_breakdown": {},
                        "error_summary": {"stream": [], "sync": []},
                        "stream_status_counts": {},
                        "sync_status_counts": {},
                    }
                }
                raw_rows["precheck"][mode] = {
                    "stream_rows": [],
                    "sync_rows": [],
                    "environment": {
                        "environment_before": None,
                        "environment_after": None,
                    },
                }
            else:
                precheck_result, case_offset = await run_mode_precheck(
                    client,
                    mode=mode,
                    cases=cases,
                    requested_model=args.requested_model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stream_requests=args.precheck_stream_requests,
                    sync_requests=args.precheck_sync_requests,
                    admin_token=args.admin_token,
                    docker_containers=docker_containers,
                    case_offset=case_offset,
                )
                summary["precheck"][mode] = {"summary": precheck_result["summary"]}
                raw_rows["precheck"][mode] = {
                    "stream_rows": precheck_result["stream_rows"],
                    "sync_rows": precheck_result["sync_rows"],
                    "environment": precheck_result["environment"],
                }
                if not precheck_result["summary"]["passed"]:
                    summary["warnings"].append(
                        f"{mode} 预检失败，先修环境再谈并发，不然这活就是拿日志当烟花看。"
                    )
                    console_log(
                        f"预检失败但继续执行 mode={mode} abort_on_precheck_fail={args.abort_on_precheck_fail}"
                    )
                    if args.abort_on_precheck_fail:
                        summary["finished_at"] = utc_now()
                        summary["passed"] = False
                        persist_reports(run_dir, summary=summary, raw_rows=raw_rows)
                        console_log(f"预检失败，提前结束 run_label={args.run_label}")
                        print(f"压测报告已输出到: {run_dir}")
                        return 1

            stages, case_offset = await run_mode_stages(
                client,
                mode=mode,
                cases=cases,
                requested_model=args.requested_model,
                temperature=args.temperature,
                top_p=args.top_p,
                concurrency_values=concurrency_values,
                warmup_requests=args.warmup_requests,
                requests_per_stage=args.requests_per_stage,
                requests_per_stage_multiplier=args.requests_per_stage_multiplier,
                cooldown_seconds=args.cooldown_seconds,
                admin_token=args.admin_token,
                docker_containers=docker_containers,
                case_offset=case_offset,
            )
            raw_rows["main"][mode] = stages
            summary["main"][mode] = [{"stage_name": stage["stage_name"], "summary": stage["summary"]} for stage in stages]

        resilience_rows, case_offset = await run_resilience_checks(
            client,
            mode="rag",
            cases=cases,
            requested_model=args.requested_model,
            temperature=args.temperature,
            top_p=args.top_p,
            admin_token=args.admin_token,
            docker_containers=docker_containers,
            case_offset=case_offset,
        )
        raw_rows["resilience"] = resilience_rows
        summary["resilience"] = [{"summary": item["summary"]} for item in resilience_rows]
        raw_rows["run_environment"]["environment_after"] = await collect_environment_snapshot(
            client,
            admin_token=args.admin_token,
            docker_containers=docker_containers,
        )

    main_passed = all(
        stage["summary"]["passed"]
        for mode in modes
        for stage in summary["main"][mode]
    )
    resilience_passed = all(item["summary"]["passed"] for item in summary["resilience"])
    precheck_passed = all(item["summary"]["passed"] for item in summary["precheck"].values())
    summary["finished_at"] = utc_now()
    summary["passed"] = precheck_passed and main_passed and resilience_passed

    if docker_containers and any(
        item.get("summary", {}).get("passed") for item in summary["resilience"]
    ):
        summary["warnings"].append(
            "容器指标已按快照采集；若要看持续 CPU/内存趋势，别偷懒，配合现成监控系统再复跑一次。"
        )
    if not docker_containers:
        summary["warnings"].append(
            "未提供 --docker-container，本次没有容器 CPU/内存/网络快照，只能看 HTTP 侧结果。"
        )
    summary["warnings"].append("只有与部署一致的 HTTP 微服务模式结果，才能拿去当上线依据。")

    persist_reports(run_dir, summary=summary, raw_rows=raw_rows)
    console_log(
        f"压测完成 passed={summary['passed']} "
        f"precheck_passed={precheck_passed} main_passed={main_passed} resilience_passed={resilience_passed}"
    )
    print(f"压测报告已输出到: {run_dir}")
    return 0 if summary["passed"] else 1


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("压测被手动中断。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"压测脚本异常退出: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
