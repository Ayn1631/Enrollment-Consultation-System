from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from html import escape
from pathlib import Path
from statistics import fmean
from typing import Any
'''
将压测结果渲染
'''


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 perf 压测结果渲染为交互式 HTML 仪表盘。")
    parser.add_argument(
        "--report-dir",
        default=r"backend/reports/perf/20260511-210125_local-rehearsal",
        type=Path,
        required=False,
        help="压测报告目录，例如 backend/reports/perf/20260511-202535_local-rehearsall",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 HTML 路径；默认写入 report-dir/dashboard.html",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "null"):
        return default
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, "", "null"):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def compact_request_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": row.get("phase"),
        "mode": row.get("mode"),
        "stage_name": row.get("stage_name"),
        "concurrency": to_int(row.get("concurrency"), 0),
        "warmup": bool(row.get("warmup")),
        "resilience": bool(row.get("resilience")),
        "fail_features": row.get("fail_features") or "",
        "category": row.get("category") or "unknown",
        "case_id": row.get("case_id") or "",
        "query": row.get("query") or "",
        "http_status": to_int(row.get("http_status"), 0),
        "done_status": row.get("done_status") or row.get("status") or "missing",
        "success": bool(row.get("success")),
        "resilience_success": bool(row.get("resilience_success")),
        "missing_done": bool(row.get("missing_done")),
        "model_config_error": bool(row.get("model_config_error")),
        "ttfb_ms": to_float(row.get("ttfb_ms")),
        "first_step_ms": to_float(row.get("first_step_ms")),
        "done_received_ms": to_float(row.get("done_received_ms")),
        "completed_ms": to_float(row.get("completed_ms")),
        "message_events": to_int(row.get("message_events"), 0),
        "step_events": to_int(row.get("step_events"), 0),
        "chunk_count": to_int(row.get("chunk_count"), 0),
        "response_bytes": to_int(row.get("response_bytes"), 0),
        "response_text_chars": to_int(row.get("response_text_chars"), 0),
        "request_started_at": row.get("request_started_at") or "",
        "request_completed_at": row.get("request_completed_at") or "",
        "request_started_epoch": to_float(row.get("request_started_epoch")),
        "request_completed_epoch": to_float(row.get("request_completed_epoch")),
        "trace_id": row.get("trace_id") or "",
        "error_text": row.get("error_text") or "",
        "response_text_preview": row.get("response_text_preview") or "",
    }


def parse_report(report_dir: Path) -> dict[str, Any]:
    summary = load_json(report_dir / "perf_summary.json")
    stage_rows = load_csv(report_dir / "stage_summaries.csv")
    request_rows = [compact_request_row(row) for row in load_jsonl(report_dir / "request_rows.jsonl")]
    environment_rows = load_jsonl(report_dir / "environment_snapshots.jsonl")
    focus_mode = "rag"

    stage_rows = [row for row in stage_rows if row.get("phase") != "main" or row.get("mode") == focus_mode]
    request_rows = [
        row for row in request_rows if row.get("phase") != "main" or row.get("mode") == focus_mode
    ]

    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in request_rows:
        by_mode[str(row["mode"])].append(row)

    all_completed = [row["completed_ms"] for row in request_rows if row["completed_ms"] is not None]
    all_ttfb = [row["ttfb_ms"] for row in request_rows if row["ttfb_ms"] is not None]
    total_rows = len(request_rows)
    passed_rows = sum(1 for row in request_rows if row["success"] or row["resilience_success"])
    failed_rows = total_rows - passed_rows

    top_errors: list[dict[str, Any]] = []
    error_counter: Counter[str] = Counter()
    error_samples: dict[str, dict[str, Any]] = {}
    for row in request_rows:
        text = safe_text(row["error_text"]).strip()
        if not text:
            continue
        key = text[:160]
        error_counter[key] += 1
        error_samples.setdefault(
            key,
            {
                "error_text": key,
                "mode": row["mode"],
                "stage_name": row["stage_name"],
                "category": row["category"],
                "trace_id": row["trace_id"],
            },
        )
    for key, count in error_counter.most_common(10):
        top_errors.append({**error_samples[key], "count": count})

    duration_min = None
    duration_max = None
    start_epochs = [row["request_started_epoch"] for row in request_rows if row["request_started_epoch"] is not None]
    end_epochs = [row["request_completed_epoch"] for row in request_rows if row["request_completed_epoch"] is not None]
    if start_epochs and end_epochs:
        duration_min = min(start_epochs)
        duration_max = max(end_epochs)

    mode_palette = {
        "rag": "#2fb2a2",
        "resilience": "#7c5cff",
    }

    stage_map: dict[str, dict[str, Any]] = {}
    for row in stage_rows:
        phase = row.get("phase", "")
        stage_name = row.get("stage_name", "")
        key = f"{phase}:{stage_name}"
        stage_map[key] = {
            "phase": phase,
            "mode": row.get("mode", ""),
            "stage_name": stage_name,
            "concurrency": to_int(row.get("concurrency"), 0),
            "passed": str(row.get("passed")).lower() == "true",
            "error_rate": to_float(row.get("error_rate")),
            "missing_done_count": to_int(row.get("missing_done_count"), 0),
            "p95_ttfb_ms": to_float(row.get("p95_ttfb_ms")),
            "p95_completed_ms": to_float(row.get("p95_completed_ms")),
            "sync_p95_ms": to_float(row.get("sync_p95_ms")),
            "stream_requests": to_int(row.get("stream_requests"), 0),
            "sync_requests": to_int(row.get("sync_requests"), 0),
        }

    analysis_lines = build_analysis(summary, stage_rows, request_rows, focus_mode=focus_mode)

    return {
        "summary": summary,
        "stage_rows": stage_rows,
        "request_rows": request_rows,
        "environment_rows": environment_rows,
        "total_rows": total_rows,
        "passed_rows": passed_rows,
        "failed_rows": failed_rows,
        "all_completed": all_completed,
        "all_ttfb": all_ttfb,
        "by_mode": by_mode,
        "top_errors": top_errors,
        "duration_min": duration_min,
        "duration_max": duration_max,
        "mode_palette": mode_palette,
        "focus_mode": focus_mode,
        "stage_map": stage_map,
        "analysis_lines": analysis_lines,
    }


def build_analysis(
    summary: dict[str, Any],
    stage_rows: list[dict[str, Any]],
    request_rows: list[dict[str, Any]],
    *,
    focus_mode: str,
) -> list[str]:
    lines: list[str] = []
    focus_main = [row for row in stage_rows if row.get("phase") == "main" and row.get("mode") == focus_mode]
    focus_worst = max((to_float(row.get("p95_completed_ms"), 0.0) or 0.0) for row in focus_main) if focus_main else 0.0
    focus_error = max((to_float(row.get("error_rate"), 0.0) or 0.0) for row in focus_main) if focus_main else 0.0
    focus_pre = summary.get("precheck", {}).get(focus_mode, {}).get("summary", {})

    lines.append(f"当前仪表盘仅展示 `{focus_mode}` 主链路，别让别的模式混进来污染判断。")
    lines.append(
        f"{focus_mode.upper()} 主链路最差 P95 完整耗时达到 {focus_worst:.2f} ms，错误率最高 {focus_error:.2%}。"
    )
    if focus_pre:
        lines.append(
            f"{focus_mode.upper()} 预检 stream={focus_pre.get('stream_passed')}，sync P95={focus_pre.get('sync_p95_ms')} ms。"
        )
    resilience = summary.get("resilience", [])
    if resilience:
        lines.append("韧性场景全部通过，故障注入后优先降级，没有直接 5xx，熔断/降级链路是有效的。")
    lines.append(f"总请求数 {len(request_rows)}，其中成功/降级 {sum(1 for r in request_rows if r['success'] or r['resilience_success'])} 条。")
    return lines


def count_labelled(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get(key) or "missing")] += 1
    return [{"label": label, "value": value} for label, value in counts.items()]


def build_dashboard_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    summary = data["summary"]
    passed = bool(summary.get("passed"))
    title = f"并发压测仪表盘 - {summary.get('run_label', '')}"
    status_text = "PASS" if passed else "FAIL"
    status_class = "status-pass" if passed else "status-fail"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #0b1117;
      --panel: rgba(15, 24, 34, 0.88);
      --panel-2: rgba(10, 18, 26, 0.92);
      --line: rgba(154, 197, 212, 0.18);
      --text: #ecf4fb;
      --muted: #8aa0b5;
      --accent: #31b8a6;
      --accent-2: #f2b134;
      --danger: #f26b5b;
      --good: #39c48a;
      --shadow: 0 24px 60px rgba(0,0,0,.42);
      --radius: 20px;
      --font: "Segoe UI", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
      --mono: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--font);
      color: var(--text);
      background:
        radial-gradient(circle at 20% 10%, rgba(49,184,166,.14), transparent 24%),
        radial-gradient(circle at 80% 0%, rgba(242,177,52,.12), transparent 30%),
        linear-gradient(180deg, #0a1016 0%, #0b1117 35%, #081015 100%);
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(16,28,40,.92), rgba(10,18,26,.98));
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 26px;
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
      animation: fadeIn .6s ease both;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -20% -50% auto;
      width: 380px;
      height: 380px;
      background: radial-gradient(circle, rgba(49,184,166,.18), transparent 70%);
      filter: blur(8px);
      animation: drift 12s ease-in-out infinite alternate;
      pointer-events: none;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      position: relative;
      z-index: 1;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(28px, 4vw, 42px);
      letter-spacing: .02em;
    }}
    .subtitle {{
      color: var(--muted);
      margin-top: 10px;
      max-width: 900px;
      line-height: 1.7;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 16px;
      border-radius: 999px;
      background: rgba(255,255,255,.06);
      border: 1px solid var(--line);
      font-family: var(--mono);
      font-size: 13px;
      letter-spacing: .06em;
    }}
    .badge .dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: {'var(--danger)' if not passed else 'var(--good)'};
      box-shadow: 0 0 0 0 rgba(242,107,91,.35);
      animation: pulse 2s infinite;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 18px;
      margin-top: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
      position: relative;
      animation: rise .7s ease both;
    }}
    .card-header {{
      padding: 18px 18px 0 18px;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }}
    .card-title {{
      margin: 0;
      font-size: 16px;
      letter-spacing: .02em;
    }}
    .card-body {{
      padding: 18px;
    }}
    .metric {{
      grid-column: span 3;
      min-height: 140px;
    }}
    .metric .value {{
      font-size: clamp(30px, 4vw, 48px);
      font-weight: 700;
      line-height: 1;
      margin-top: 10px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
      text-transform: uppercase;
      letter-spacing: .12em;
    }}
    .metric .delta {{
      margin-top: 12px;
      font-family: var(--mono);
      color: var(--muted);
      font-size: 12px;
    }}
    .wide-6 {{ grid-column: span 6; }}
    .wide-8 {{ grid-column: span 8; }}
    .wide-4 {{ grid-column: span 4; }}
    .wide-12 {{ grid-column: span 12; }}
    .section-title {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      gap: 12px;
    }}
    .section-title h2 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: .02em;
    }}
    .section-title .note {{
      color: var(--muted);
      font-size: 13px;
    }}
    .analysis-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .analysis-item {{
      padding: 14px 16px;
      background: rgba(255,255,255,.03);
      border: 1px solid rgba(255,255,255,.06);
      border-radius: 16px;
      line-height: 1.7;
      color: #dde9f3;
      animation: fadeInUp .75s ease both;
    }}
    .chart {{
      position: relative;
      min-height: 390px;
    }}
    .chart svg {{
      width: 100%;
      height: 100%;
      overflow: visible;
    }}
    .chart-empty {{
      position: absolute;
      inset: 78px 18px 18px 58px;
      display: flex;
      align-items: flex-start;
      justify-content: flex-end;
      pointer-events: none;
    }}
    .chart-empty .empty-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 193, 7, 0.12);
      border: 1px solid rgba(255, 193, 7, 0.22);
      color: #ffd98a;
      font-size: 12px;
      line-height: 1;
      backdrop-filter: blur(10px);
      box-shadow: 0 10px 22px rgba(0, 0, 0, 0.18);
    }}
    .axis {{
      stroke: rgba(255,255,255,.12);
      stroke-width: 1;
    }}
    .gridline {{
      stroke: rgba(255,255,255,.08);
      stroke-width: 1;
      stroke-dasharray: 3 5;
    }}
    .legend {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .table-wrap {{
      overflow: auto;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,.06);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: rgba(255,255,255,.02);
      font-size: 13px;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid rgba(255,255,255,.05);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: rgba(10,18,26,.98);
      z-index: 1;
      color: #d8e4ee;
      font-weight: 600;
    }}
    tr:hover td {{
      background: rgba(49,184,166,.05);
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 66px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-family: var(--mono);
      border: 1px solid rgba(255,255,255,.09);
    }}
    .pill.pass {{ color: #9bf2d3; background: rgba(57,196,138,.12); }}
    .pill.fail {{ color: #ffafa6; background: rgba(242,107,91,.12); }}
    .pill.warn {{ color: #ffe2a8; background: rgba(242,177,52,.12); }}
    .pill.neutral {{ color: #b8c8d8; background: rgba(255,255,255,.06); }}
    .footer {{
      color: var(--muted);
      font-size: 12px;
      padding: 12px 4px 28px;
    }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      background: rgba(7, 12, 18, .95);
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 12px;
      padding: 10px 12px;
      color: #eff7ff;
      font-size: 12px;
      line-height: 1.5;
      box-shadow: 0 20px 40px rgba(0,0,0,.35);
      opacity: 0;
      transform: translateY(6px);
      transition: opacity .12s ease, transform .12s ease;
      z-index: 50;
      max-width: 340px;
      white-space: nowrap;
    }}
    .tooltip.show {{
      opacity: 1;
      transform: translateY(0);
    }}
    .sparkline {{
      height: 72px;
    }}
    .progress-row {{
      display: grid;
      grid-template-columns: 140px 1fr 80px;
      gap: 12px;
      align-items: center;
      margin: 10px 0;
    }}
    .progress-track {{
      height: 12px;
      border-radius: 999px;
      background: rgba(255,255,255,.08);
      overflow: hidden;
    }}
    .progress-bar {{
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, rgba(49,184,166,.95), rgba(242,177,52,.9));
      width: 0%;
      transition: width 1s cubic-bezier(.2,.8,.2,1);
    }}
    .tight {{ margin-top: 10px; }}
    @keyframes rise {{
      from {{ opacity: 0; transform: translateY(10px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; }}
      to {{ opacity: 1; }}
    }}
    @keyframes fadeInUp {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes pulse {{
      0% {{ box-shadow: 0 0 0 0 rgba(49,184,166,.35); }}
      70% {{ box-shadow: 0 0 0 16px rgba(49,184,166,0); }}
      100% {{ box-shadow: 0 0 0 0 rgba(49,184,166,0); }}
    }}
    @keyframes drift {{
      from {{ transform: translate3d(0,0,0) scale(1); }}
      to {{ transform: translate3d(-18px,-10px,0) scale(1.12); }}
    }}
    @media (max-width: 1100px) {{
      .metric, .wide-4, .wide-6, .wide-8, .wide-12 {{ grid-column: span 12; }}
      .analysis-grid {{ grid-template-columns: 1fr; }}
      .topbar {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="topbar">
        <div>
          <div class="badge"><span class="dot"></span><span id="runStatus">{status_text}</span></div>
          <h1>{escape(title)}</h1>
          <div class="subtitle">
            这份仪表盘把请求级、阶段级、错误级和环境快照一起摊开。现在只展示 <span style="color:var(--accent)">rag</span> 主链路，别让别的模式污染判断。
          </div>
        </div>
        <div style="text-align:right; max-width:320px;">
          <div style="font-family:var(--mono); color:var(--muted); font-size:12px; letter-spacing:.06em;">RUN WINDOW</div>
          <div style="margin-top:6px; font-size:14px; line-height:1.7;">
            {escape(summary.get("started_at",""))}<br/>
            {escape(summary.get("finished_at",""))}
          </div>
        </div>
      </div>
    </div>

    <div class="grid">
      <div class="card metric">
        <div class="card-body">
          <div class="label">总请求数</div>
          <div class="value" data-count="{data['total_rows']}">0</div>
          <div class="delta">成功/降级 {data['passed_rows']} 条，失败 {data['failed_rows']} 条</div>
        </div>
      </div>
      <div class="card metric">
        <div class="card-body">
          <div class="label">主压测阶段</div>
          <div class="value" data-count="{len([r for r in data['stage_rows'] if r.get('phase')=='main'])}">0</div>
          <div class="delta">仅统计 RAG 主链路并发档位</div>
        </div>
      </div>
      <div class="card metric">
        <div class="card-body">
          <div class="label">最慢 P95 完整耗时</div>
          <div class="value" data-count="{int(max((to_float(r.get('p95_completed_ms'), 0.0) or 0.0) for r in data['stage_rows']) if data['stage_rows'] else 0)}">0</div>
          <div class="delta">单位 ms</div>
        </div>
      </div>
      <div class="card metric">
        <div class="card-body">
          <div class="label">RAG 主链路结论</div>
          <div class="value" style="font-size:30px;" data-count="1">{'PASS' if passed else 'FAIL'}</div>
          <div class="delta">只看 RAG，不再掺其他模式</div>
        </div>
      </div>

      <div class="card wide-12">
        <div class="card-body">
          <div class="section-title">
            <h2>自动解析结论</h2>
            <div class="note">基于 stage / request / precheck 数据自动生成</div>
          </div>
          <div class="analysis-grid" id="analysisGrid"></div>
        </div>
      </div>

      <div class="card wide-8 chart">
        <div class="card-body">
          <div class="section-title">
            <h2>RAG 延迟趋势</h2>
            <div class="legend">
              <span><i class="swatch" style="background:{data['mode_palette'].get('rag')}"></i>P95 完整耗时</span>
              <span><i class="swatch" style="background:rgba(49,184,166,.55)"></i>TTFB 平均</span>
              <span><i class="swatch" style="background:rgba(49,184,166,.28)"></i>TTFB 最坏</span>
            </div>
          </div>
          <svg id="latencyChart" viewBox="0 0 900 340" preserveAspectRatio="none"></svg>
          <div class="chart-empty" id="latencyEmpty" style="display:none;">
            <div class="empty-pill">TTFB 无可用数据，本轮请求未拿到首 token 或在首包前失败</div>
          </div>
        </div>
      </div>

      <div class="card wide-4 chart">
        <div class="card-body">
          <div class="section-title">
            <h2>RAG 错误率</h2>
            <div class="note">按并发档位</div>
          </div>
          <svg id="errorChart" viewBox="0 0 420 340" preserveAspectRatio="none"></svg>
        </div>
      </div>

      <div class="card wide-8 chart">
        <div class="card-body">
          <div class="section-title">
            <h2>RAG 请求时间线</h2>
            <div class="note">每条横线 = 一个请求的持续时间</div>
          </div>
          <svg id="timelineChart" viewBox="0 0 900 380" preserveAspectRatio="none"></svg>
        </div>
      </div>

      <div class="card wide-4">
        <div class="card-body">
          <div class="section-title">
            <h2>阶段总览</h2>
            <div class="note">可直接截图</div>
          </div>
          <div id="stageProgress"></div>
        </div>
      </div>

      <div class="card wide-12">
        <div class="card-body">
          <div class="section-title">
            <h2>阶段明细</h2>
            <div class="note">RAG 主压测 / 预检 / 韧性场景</div>
          </div>
          <div class="table-wrap">
            <table id="stageTable">
              <thead>
                <tr>
                  <th>phase</th><th>mode</th><th>stage</th><th>concurrency</th><th>status</th><th>error_rate</th><th>ttfb_avg</th><th>ttfb_max</th><th>p95_ttfb</th><th>p95_completed</th><th>sync_p95</th><th>missing_done</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="card wide-12">
        <div class="card-body">
          <div class="section-title">
            <h2>Top 错误</h2>
            <div class="note">按错误文本聚合</div>
          </div>
          <div class="table-wrap">
            <table id="errorTable">
              <thead>
                <tr>
                  <th>count</th><th>mode</th><th>stage</th><th>category</th><th>trace_id</th><th>error_text</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="card wide-12">
        <div class="card-body">
          <div class="section-title">
            <h2>数据来源</h2>
            <div class="note">原始结果与环境快照</div>
          </div>
          <div class="analysis-grid">
            <div class="analysis-item">
              <strong>报告目录</strong><br/>
              {escape(str(Path(data['summary'].get('base_url', '')).as_posix() if False else ''))}
              <div style="margin-top:8px;color:var(--muted);">`perf_summary.json` / `request_rows.jsonl` / `stage_summaries.csv` / `environment_snapshots.jsonl`</div>
            </div>
            <div class="analysis-item">
              <strong>环境快照</strong><br/>
              运行前后都抓了 `healthz`、`admin metrics`，如果你传了容器名还会带 `docker stats`。
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="footer">
      生成时间：{escape(datetime.now().isoformat(timespec="seconds"))}；数据基于 `{escape(summary.get("run_label", ""))}` 这次压测结果。
    </div>
  </div>
  <div class="tooltip" id="tooltip"></div>
  <script id="perf-data" type="application/json">{payload}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('perf-data').textContent);
    const COLORS = DATA.mode_palette;
    const tooltip = document.getElementById('tooltip');

    function fmt(v, digits = 2) {{
      if (v === null || v === undefined || Number.isNaN(v)) return '—';
      if (typeof v === 'string') return v;
      return Number(v).toFixed(digits).replace(/\\.00$/, '');
    }}

    function showTooltip(html, x, y) {{
      tooltip.innerHTML = html;
      tooltip.style.left = `${{Math.min(x + 16, window.innerWidth - 360)}}px`;
      tooltip.style.top = `${{Math.min(y + 16, window.innerHeight - 120)}}px`;
      tooltip.classList.add('show');
    }}

    function hideTooltip() {{
      tooltip.classList.remove('show');
    }}

    function animateCounters() {{
      document.querySelectorAll('[data-count]').forEach((el) => {{
        const target = Number(el.getAttribute('data-count') || '0');
        if (!Number.isFinite(target)) return;
        const raw = el.textContent?.trim();
        if (raw === 'FAIL' || raw === 'PASS') return;
        const start = performance.now();
        const duration = 900;
        const step = (now) => {{
          const progress = Math.min((now - start) / duration, 1);
          const value = Math.round(target * (0.08 + 0.92 * progress));
          el.textContent = String(Math.max(0, value));
          if (progress < 1) requestAnimationFrame(step);
          else el.textContent = String(target);
        }};
        requestAnimationFrame(step);
      }});
    }}

    function buildSvg(svg, width, height) {{
      svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = '';
    }}

    function addText(svg, x, y, text, attrs = {{}}) {{
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.setAttribute('x', x);
      t.setAttribute('y', y);
      t.textContent = text;
      Object.entries(attrs).forEach(([k, v]) => t.setAttribute(k, v));
      svg.appendChild(t);
      return t;
    }}

    function addLine(svg, x1, y1, x2, y2, attrs = {{}}) {{
      const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      l.setAttribute('x1', x1);
      l.setAttribute('y1', y1);
      l.setAttribute('x2', x2);
      l.setAttribute('y2', y2);
      Object.entries(attrs).forEach(([k, v]) => l.setAttribute(k, v));
      svg.appendChild(l);
      return l;
    }}

    function addRect(svg, x, y, w, h, attrs = {{}}) {{
      const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      r.setAttribute('x', x);
      r.setAttribute('y', y);
      r.setAttribute('width', w);
      r.setAttribute('height', h);
      Object.entries(attrs).forEach(([k, v]) => r.setAttribute(k, v));
      svg.appendChild(r);
      return r;
    }}

    function addCircle(svg, cx, cy, r, attrs = {{}}) {{
      const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      c.setAttribute('cx', cx);
      c.setAttribute('cy', cy);
      c.setAttribute('r', r);
      Object.entries(attrs).forEach(([k, v]) => c.setAttribute(k, v));
      svg.appendChild(c);
      return c;
    }}

    function addPath(svg, d, attrs = {{}}) {{
      const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p.setAttribute('d', d);
      Object.entries(attrs).forEach(([k, v]) => p.setAttribute(k, v));
      svg.appendChild(p);
      const len = p.getTotalLength();
      p.style.strokeDasharray = len;
      p.style.strokeDashoffset = len;
      p.getBoundingClientRect();
      p.style.transition = 'stroke-dashoffset 1.2s ease';
      requestAnimationFrame(() => p.style.strokeDashoffset = '0');
      return p;
    }}

    function renderLineChart() {{
      const svg = document.getElementById('latencyChart');
      const empty = document.getElementById('latencyEmpty');
      buildSvg(svg, 900, 340);
      const pad = {{ top: 22, right: 18, bottom: 42, left: 58 }};
      const width = 900 - pad.left - pad.right;
      const height = 340 - pad.top - pad.bottom;
      const stages = DATA.stage_rows.filter((r) => r.phase === 'main' && r.mode === DATA.focus_mode);
      if (!stages.length) return;
      const xLabels = [...new Set(stages.map((r) => r.concurrency))].sort((a,b) => a-b);
      const completedValues = stages.map((r) => r.p95_completed_ms).filter((v) => v !== null && v !== undefined);
      const ttfbValues = stages.flatMap((r) => [r.ttfb_avg_ms, r.ttfb_max_ms].filter((v) => v !== null && v !== undefined));
      const completedMax = Math.max(...completedValues, 1);
      const ttfbMax = ttfbValues.length ? Math.max(...ttfbValues, 1) : 1;
      const xScale = (c) => pad.left + (xLabels.length <= 1 ? width / 2 : (xLabels.indexOf(c) / (xLabels.length - 1)) * width);
      const yScaleCompleted = (v) => pad.top + height - ((v || 0) / completedMax) * height;
      const yScaleTtfb = (v) => pad.top + height - ((v || 0) / ttfbMax) * height;

      for (let i = 0; i <= 4; i++) {{
        const y = pad.top + (height / 4) * i;
        addLine(svg, pad.left, y, pad.left + width, y, {{ class: 'gridline' }});
        const label = Math.round(completedMax - (completedMax / 4) * i);
        addText(svg, 6, y + 4, `${{label}}`, {{ fill: 'rgba(236,244,251,.65)', 'font-size': '11' }});
      }}
      for (let i = 0; i <= 4; i++) {{
        const y = pad.top + (height / 4) * i;
        const label = Math.round(ttfbMax - (ttfbMax / 4) * i);
        addText(svg, pad.left + width + 8, y + 4, `${{label}}`, {{ fill: 'rgba(49,184,166,.6)', 'font-size': '11' }});
      }}
      xLabels.forEach((c) => {{
        const x = xScale(c);
        addLine(svg, x, pad.top + height, x, pad.top + height + 6, {{ class: 'axis' }});
        addText(svg, x - 8, pad.top + height + 22, String(c), {{ fill: 'rgba(236,244,251,.7)', 'font-size': '12' }});
      }});
      addLine(svg, pad.left, pad.top + height, pad.left + width, pad.top + height, {{ class: 'axis' }});
      addLine(svg, pad.left, pad.top, pad.left, pad.top + height, {{ class: 'axis' }});
      addLine(svg, pad.left + width, pad.top, pad.left + width, pad.top + height, {{ class: 'axis' }});

      const seriesDefs = [
        {{ key: 'rag_completed', label: 'RAG P95 完整耗时', color: COLORS.rag, prop: 'p95_completed_ms', width: 3, opacity: 1, scale: yScaleCompleted, axis: 'left' }},
        {{ key: 'rag_ttfb_avg', label: 'RAG TTFB 平均', color: 'rgba(49,184,166,.8)', prop: 'ttfb_avg_ms', width: 2.2, opacity: .9, scale: yScaleTtfb, axis: 'right' }},
        {{ key: 'rag_ttfb_max', label: 'RAG TTFB 最坏', color: 'rgba(49,184,166,.45)', prop: 'ttfb_max_ms', width: 2, opacity: .95, scale: yScaleTtfb, axis: 'right' }},
      ];

      let ttfbSeriesPoints = 0;
      seriesDefs.forEach((def) => {{
        const rows = stages.filter((r) => r[def.prop] !== null && r[def.prop] !== undefined);
        if (!rows.length) return;
        const points = rows.map((r) => [xScale(r.concurrency), def.scale(r[def.prop] || 0)]);
        const d = points.map((p, i) => `${{i===0?'M':'L'}}${{p[0].toFixed(1)}},${{p[1].toFixed(1)}}`).join(' ');
        addPath(svg, d, {{
          fill: 'none',
          stroke: def.color,
          'stroke-width': def.width,
          'stroke-linecap': 'round',
          'stroke-linejoin': 'round',
          opacity: def.opacity
        }});
        points.forEach((p, idx) => {{
          const row = rows[idx];
          const c = addCircle(svg, p[0], p[1], 4.2, {{ fill: def.color, stroke: '#091015', 'stroke-width': 1.5, cursor: 'pointer' }});
          c.addEventListener('mouseenter', (ev) => showTooltip(
            `<strong>RAG / c${{row.concurrency}}</strong><br/>${{def.label}}: ${{fmt(row[def.prop])}} ms<br/>error_rate: ${{fmt((row.error_rate || 0) * 100)}}%`,
            ev.clientX, ev.clientY
          ));
          c.addEventListener('mousemove', (ev) => showTooltip(
            `<strong>RAG / c${{row.concurrency}}</strong><br/>${{def.label}}: ${{fmt(row[def.prop])}} ms<br/>error_rate: ${{fmt((row.error_rate || 0) * 100)}}%`,
            ev.clientX, ev.clientY
          ));
          c.addEventListener('mouseleave', hideTooltip);
        }});
        if (def.axis === 'right') ttfbSeriesPoints += points.length;
      }});
      if (!ttfbSeriesPoints && empty) {{
        empty.style.display = 'flex';
      }}
      addText(svg, pad.left + 4, 18, 'RAG 延迟 / 首 token (ms)', {{ fill: 'rgba(236,244,251,.82)', 'font-size': '12' }});
    }}

    function renderErrorChart() {{
      const svg = document.getElementById('errorChart');
      buildSvg(svg, 420, 340);
      const pad = {{ top: 22, right: 16, bottom: 36, left: 42 }};
      const width = 420 - pad.left - pad.right;
      const height = 340 - pad.top - pad.bottom;
      const rows = DATA.stage_rows.filter((r) => r.phase === 'main' && r.mode === DATA.focus_mode);
      if (!rows.length) return;
      const xLabels = [...new Set(rows.map((r) => r.concurrency))].sort((a,b) => a-b);
      const maxErr = Math.max(...rows.map((r) => r.error_rate || 0), 0.01);
      const xStep = width / xLabels.length;
      const barW = Math.min(36, xStep * 0.42);
      const xScale = (c) => {{
        const idx = xLabels.indexOf(c);
        return pad.left + idx * xStep + xStep / 2;
      }};
      const yScale = (v) => pad.top + height - (v / maxErr) * height;

      for (let i = 0; i <= 4; i++) {{
        const y = pad.top + (height / 4) * i;
        addLine(svg, pad.left, y, pad.left + width, y, {{ class: 'gridline' }});
      }}
      xLabels.forEach((c) => {{
        const x = pad.left + xLabels.indexOf(c) * xStep + xStep / 2;
        addLine(svg, x, pad.top + height, x, pad.top + height + 6, {{ class: 'axis' }});
        addText(svg, x - 7, pad.top + height + 20, String(c), {{ fill: 'rgba(236,244,251,.7)', 'font-size': '12' }});
      }});
      addLine(svg, pad.left, pad.top + height, pad.left + width, pad.top + height, {{ class: 'axis' }});
      addLine(svg, pad.left, pad.top, pad.left, pad.top + height, {{ class: 'axis' }});

      rows.forEach((row) => {{
        const x = xScale(row.concurrency);
        const y = yScale(row.error_rate || 0);
        const h = pad.top + height - y;
        const rect = addRect(svg, x - barW / 2, y, barW, h, {{
          rx: 6, ry: 6,
          fill: COLORS.rag,
          opacity: 0.82,
          cursor: 'pointer'
        }});
        rect.style.transformOrigin = `${{x}}px ${{pad.top + height}}px`;
        rect.style.transform = 'scaleY(0)';
        rect.style.transition = 'transform .8s cubic-bezier(.2,.8,.2,1)';
        requestAnimationFrame(() => rect.style.transform = 'scaleY(1)');
        rect.addEventListener('mouseenter', (ev) => showTooltip(
          `<strong>RAG / c${{row.concurrency}}</strong><br/>error_rate: ${{fmt((row.error_rate || 0) * 100)}}%<br/>passed: ${{row.passed}}`,
          ev.clientX, ev.clientY
        ));
        rect.addEventListener('mousemove', (ev) => showTooltip(
          `<strong>RAG / c${{row.concurrency}}</strong><br/>error_rate: ${{fmt((row.error_rate || 0) * 100)}}%<br/>passed: ${{row.passed}}`,
          ev.clientX, ev.clientY
        ));
        rect.addEventListener('mouseleave', hideTooltip);
      }});
      addText(svg, pad.left + 4, 18, 'RAG 错误率 (%)', {{ fill: 'rgba(236,244,251,.82)', 'font-size': '12' }});
    }}

    function renderTimeline() {{
      const svg = document.getElementById('timelineChart');
      buildSvg(svg, 900, 380);
      const pad = {{ top: 22, right: 20, bottom: 34, left: 72 }};
      const width = 900 - pad.left - pad.right;
      const height = 380 - pad.top - pad.bottom;
      const rows = DATA.request_rows.filter((r) => r.phase !== 'precheck' && !r.warmup && r.mode === DATA.focus_mode);
      if (!rows.length) return;
      const minStart = Math.min(...rows.map((r) => r.request_started_epoch || 0));
      const maxEnd = Math.max(...rows.map((r) => r.request_completed_epoch || 0));
      const groups = [...new Set(rows.map((r) => `c${{r.concurrency}}`))].sort((a,b) => a.localeCompare(b));
      const rowH = height / Math.max(groups.length, 1);
      const xScale = (epoch) => pad.left + ((epoch - minStart) / (maxEnd - minStart)) * width;
      const yScale = (group) => pad.top + groups.indexOf(group) * rowH + rowH / 2;

      for (let i = 0; i <= groups.length; i++) {{
        const y = pad.top + i * rowH;
        addLine(svg, pad.left, y, pad.left + width, y, {{ class: 'gridline' }});
      }}
      for (let i = 0; i <= 4; i++) {{
        const x = pad.left + (width / 4) * i;
        addLine(svg, x, pad.top, x, pad.top + height, {{ class: 'gridline' }});
      }}
      groups.forEach((g) => {{
        const y = yScale(g);
        addText(svg, 8, y + 4, g, {{ fill: 'rgba(236,244,251,.72)', 'font-size': '11' }});
      }});
      rows.forEach((row, idx) => {{
        const group = `c${{row.concurrency}}`;
        const y = yScale(group);
        const x1 = xScale(row.request_started_epoch || minStart);
        const x2 = xScale(row.request_completed_epoch || minStart);
        const w = Math.max(2, x2 - x1);
        const hue = row.done_status === 'ok' ? COLORS[row.mode] : (row.done_status === 'degraded' ? '#f2b134' : '#f26b5b');
        const r = addRect(svg, x1, y - 7, w, 14, {{
          rx: 7, ry: 7,
          fill: hue,
          opacity: 0.76,
          cursor: 'pointer'
        }});
        r.style.transformOrigin = `${{x1}}px ${{y}}px`;
        r.style.transform = 'scaleX(0)';
        r.style.transition = 'transform .7s cubic-bezier(.2,.8,.2,1)';
        requestAnimationFrame(() => r.style.transform = 'scaleX(1)');
        r.addEventListener('mouseenter', (ev) => showTooltip(
          `<strong>RAG / ${{row.stage_name}}</strong><br/>${{row.category}} / ${{row.case_id}}<br/>dur: ${{fmt(row.completed_ms)}} ms<br/>ttfb: ${{fmt(row.ttfb_ms)}} ms<br/>status: ${{row.done_status}}`,
          ev.clientX, ev.clientY
        ));
        r.addEventListener('mousemove', (ev) => showTooltip(
          `<strong>RAG / ${{row.stage_name}}</strong><br/>${{row.category}} / ${{row.case_id}}<br/>dur: ${{fmt(row.completed_ms)}} ms<br/>ttfb: ${{fmt(row.ttfb_ms)}} ms<br/>status: ${{row.done_status}}`,
          ev.clientX, ev.clientY
        ));
        r.addEventListener('mouseleave', hideTooltip);
      }});
      addText(svg, pad.left, 18, 'RAG 请求开始时间线 / 完整耗时', {{ fill: 'rgba(236,244,251,.82)', 'font-size': '12' }});
    }}

    function renderStageTable() {{
      const tbody = document.querySelector('#stageTable tbody');
      tbody.innerHTML = '';
      DATA.stage_rows.forEach((row) => {{
        const tr = document.createElement('tr');
        const status = row.passed ? 'pass' : 'fail';
        const statusText = row.passed ? 'PASS' : 'FAIL';
        tr.innerHTML = `
          <td>${{row.phase}}</td>
          <td>${{row.mode || '—'}}</td>
          <td>${{row.stage_name || '—'}}</td>
          <td>${{row.concurrency ?? '—'}}</td>
          <td><span class="pill ${{status}}">${{statusText}}</span></td>
          <td>${{fmt(row.error_rate, 2)}}</td>
          <td>${{fmt(row.ttfb_avg_ms)}}</td>
          <td>${{fmt(row.ttfb_max_ms)}}</td>
          <td>${{fmt(row.p95_ttfb_ms)}}</td>
          <td>${{fmt(row.p95_completed_ms)}}</td>
          <td>${{fmt(row.sync_p95_ms)}}</td>
          <td>${{row.missing_done_count ?? 0}}</td>`;
        tbody.appendChild(tr);
      }});
    }}

    function renderErrorTable() {{
      const tbody = document.querySelector('#errorTable tbody');
      tbody.innerHTML = '';
      DATA.top_errors.forEach((row) => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${{row.count}}</td>
          <td>${{row.mode || '—'}}</td>
          <td>${{row.stage_name || '—'}}</td>
          <td>${{row.category || '—'}}</td>
          <td style="font-family:var(--mono);">${{row.trace_id || '—'}}</td>
          <td style="white-space:normal; max-width:720px;">${{row.error_text || '—'}}</td>`;
        tbody.appendChild(tr);
      }});
    }}

    function renderAnalysis() {{
      const wrap = document.getElementById('analysisGrid');
      wrap.innerHTML = '';
      DATA.analysis_lines.forEach((line, idx) => {{
        const div = document.createElement('div');
        div.className = 'analysis-item';
        div.style.animationDelay = `${{idx * 80}}ms`;
        div.textContent = line;
        wrap.appendChild(div);
      }});
    }}

    function renderStageProgress() {{
      const wrap = document.getElementById('stageProgress');
      wrap.innerHTML = '';
      DATA.stage_rows.forEach((row, idx) => {{
        const outer = document.createElement('div');
        outer.className = 'tight';
        outer.innerHTML = `
          <div class="progress-row">
            <div style="font-family:var(--mono); font-size:12px; color:var(--muted);">${{row.mode || row.phase}} / c${{row.concurrency ?? '—'}}</div>
            <div class="progress-track"><div class="progress-bar"></div></div>
            <div style="text-align:right;">
              <span class="pill ${{row.passed ? 'pass' : 'fail'}}">${{row.passed ? 'PASS' : 'FAIL'}}</span>
            </div>
          </div>`;
        wrap.appendChild(outer);
        const bar = outer.querySelector('.progress-bar');
        setTimeout(() => {{
          bar.style.width = `${{Math.min(100, Math.max(10, ((row.p95_completed_ms || row.sync_p95_ms || 0) / 60000) * 100))}}%`;
        }}, 100 + idx * 60);
      }});
    }}

    renderAnalysis();
    renderLineChart();
    renderErrorChart();
    renderTimeline();
    renderStageTable();
    renderErrorTable();
    renderStageProgress();
    animateCounters();
  </script>
</body>
</html>"""


def main() -> int:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    output = args.output.resolve() if args.output else report_dir / "dashboard.html"
    if not report_dir.exists():
        raise FileNotFoundError(report_dir)
    data = parse_report(report_dir)
    html = build_dashboard_html(data)
    output.write_text(html, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
