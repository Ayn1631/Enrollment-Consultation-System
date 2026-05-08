from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.services.ai_stack import build_langchain_mcp_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 MCP 配置并输出已加载工具列表")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="显式指定 MCP 配置文件路径，默认优先使用 backend/mcp.json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出检测结果",
    )
    args = parser.parse_args()

    settings = _build_settings(args.config)
    runtime = asyncio.run(build_langchain_mcp_runtime(settings))
    try:
        payload = _build_report(settings=settings, runtime=runtime)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _print_report(payload)
        return 0 if payload["ok"] else 1
    finally:
        asyncio.run(runtime.aclose())


def _build_settings(config_path: Path | None) -> Settings:
    resolved = config_path
    if resolved is None:
        default_path = ROOT / "mcp.json"
        if default_path.exists():
            resolved = default_path
    settings = Settings()
    if resolved is None:
        return settings
    return settings.model_copy(update={"mcp_config_path": str(resolved.resolve())})


def _build_report(*, settings: Settings, runtime: Any) -> dict[str, Any]:
    tool_rows = []
    for item in runtime.tools:
        tool_rows.append(
            {
                "name": _tool_name(item),
                "description": _tool_description(item),
            }
        )
    server_rows = [
        {
            "alias": item.alias,
            "original_name": item.original_name,
            "transport": item.transport,
            "command": item.command,
            "url": item.url,
        }
        for item in runtime.servers
    ]
    config_path = settings.resolve_mcp_config_path()
    return {
        "ok": bool(runtime.tools),
        "config_path": str(config_path) if config_path else "",
        "server_count": len(runtime.servers),
        "tool_count": len(runtime.tools),
        "servers": server_rows,
        "tools": tool_rows,
        "notes": list(runtime.notes),
    }


def _tool_name(tool: Any) -> str:
    bound = getattr(tool, "bound", None)
    if getattr(tool, "name", None):
        return str(tool.name)
    if getattr(bound, "name", None):
        return str(bound.name)
    return "unknown_tool"


def _tool_description(tool: Any) -> str:
    bound = getattr(tool, "bound", None)
    if getattr(tool, "description", None):
        return str(tool.description)
    if getattr(bound, "description", None):
        return str(bound.description)
    return ""


def _print_report(payload: dict[str, Any]) -> None:
    print(f"MCP 配置文件: {payload['config_path'] or '未解析到'}")
    print(f"Server 数量: {payload['server_count']}")
    print(f"Tool 数量: {payload['tool_count']}")
    print("Server 列表:")
    for item in payload["servers"]:
        location = item["url"] or item["command"] or "-"
        print(f"- {item['original_name']} -> {item['alias']} [{item['transport']}] {location}")
    print("Tool 列表:")
    for item in payload["tools"]:
        description = item["description"].strip()
        if description:
            print(f"- {item['name']}: {description}")
        else:
            print(f"- {item['name']}")
    if payload["notes"]:
        print("运行说明:")
        for item in payload["notes"]:
            print(f"- {item}")
    print("检测结果: " + ("成功" if payload["ok"] else "失败"))


if __name__ == "__main__":
    raise SystemExit(main())
