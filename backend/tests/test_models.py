from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from app.models import ChatRequest


def _base_request() -> dict:
    return {
        "session_id": uuid.uuid4().hex,
        "messages": [{"role": "user", "content": "请介绍招生政策"}],
        "mode": "chat",
        "stream": True,
    }


def test_chat_request_tools_mapping_and_dedup():
    payload = _base_request()
    payload["features"] = ["rag"]
    payload["tools"] = ["react", "react", "search"]
    req = ChatRequest(**payload)
    assert req.features == ["rag", "skill_exec", "web_search"]


def test_chat_request_dependency_expansion_for_saved_skill():
    payload = _base_request()
    payload["features"] = ["use_saved_skill"]
    payload["saved_skill_id"] = "admission_faq_v1"
    req = ChatRequest(**payload)
    assert "use_saved_skill" in req.features
    assert "skill_exec" in req.features


def test_chat_request_dependency_expansion_for_citation_guard():
    payload = _base_request()
    payload["features"] = ["citation_guard"]
    req = ChatRequest(**payload)
    assert "citation_guard" in req.features
    assert "rag" in req.features


def test_chat_request_requires_saved_skill_id():
    payload = _base_request()
    payload["features"] = ["use_saved_skill"]
    with pytest.raises(ValidationError):
        ChatRequest(**payload)


def test_chat_request_agent_strategy_defaults_to_speed():
    req = ChatRequest(**_base_request())
    assert req.agent_strategy == "speed"


def test_chat_request_accepts_quality_agent_strategy():
    payload = _base_request()
    payload["agent_strategy"] = "quality"
    req = ChatRequest(**payload)
    assert req.agent_strategy == "quality"


def test_chat_mode_defaults_to_plain_dialogue_without_rag():
    req = ChatRequest(**_base_request())
    assert req.features == []


def test_rag_mode_defaults_to_rag_and_citation_guard():
    payload = _base_request()
    payload["mode"] = "rag"
    req = ChatRequest(**payload)
    assert req.features == ["rag", "citation_guard"]


def test_agent_mode_defaults_to_full_feature_chain():
    payload = _base_request()
    payload["mode"] = "agent"
    req = ChatRequest(**payload)
    assert req.features == ["rag", "skill_exec", "citation_guard"]
