from __future__ import annotations

import textwrap
from typing import Any

import httpx

from app.config import Settings


class GenerationService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(
        self,
        user_query: str,
        context_blocks: list[str],
        feature_notes: list[str],
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        if self.settings.use_mock_generation or not self.settings.api_key:
            return self._mock_generate(user_query, context_blocks, feature_notes)
        return self._remote_generate(
            user_query=user_query,
            context_blocks=context_blocks,
            feature_notes=feature_notes,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )

    def _mock_generate(self, user_query: str, context_blocks: list[str], feature_notes: list[str]) -> str:
        excerpt = "\n".join(f"- {line[:110]}" for line in context_blocks[:4]) if context_blocks else "- 未命中可靠证据。"
        notes = "\n".join(f"- {note}" for note in feature_notes) if feature_notes else "- 未启用额外增强功能。"
        return textwrap.dedent(
            f"""
            问题：{user_query}

            基于当前检索到的材料，先给你结论再给依据：
            {excerpt}

            本轮能力执行情况：
            {notes}
            """
        ).strip()

    def _remote_generate(
        self,
        user_query: str,
        context_blocks: list[str],
        feature_notes: list[str],
        model: str | None,
        temperature: float | None,
        top_p: float | None,
    ) -> str:
        context_text = "\n".join(f"- {item}" for item in context_blocks[:6]) or "- 无可靠检索证据"
        note_text = "\n".join(f"- {item}" for item in feature_notes) or "- 无"

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是中原工学院招生咨询助手。必须基于证据回答，若证据不足请明确说明不确定，"
                    "并建议联系官方招生办。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{user_query}\n\n证据：\n{context_text}\n\n"
                    f"执行备注：\n{note_text}\n\n请给出简明回答。"
                ),
            },
        ]
        payload: dict[str, Any] = {
            "model": model or "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.4 if temperature is None else temperature,
            "top_p": 0.9 if top_p is None else top_p,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            response = client.post(self.settings.api_url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError("generation response has no choices")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not content:
            raise RuntimeError("generation response content is empty")
        return str(content)

