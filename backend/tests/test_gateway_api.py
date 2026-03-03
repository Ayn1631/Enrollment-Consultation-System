from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.main import app


def _base_payload() -> dict:
    return {
        "session_id": uuid.uuid4().hex,
        "messages": [{"role": "user", "content": "请介绍招生章程重点"}],
        "mode": "chat",
        "stream": True,
        "features": ["rag", "citation_guard"],
        "strict_citation": True,
    }


def test_features_endpoint_returns_defaults():
    client = TestClient(app)
    res = client.get("/api/features")
    assert res.status_code == 200
    body = res.json()
    ids = {item["id"] for item in body}
    assert "rag" in ids
    assert "citation_guard" in ids


def test_saved_skills_endpoint_returns_list():
    client = TestClient(app)
    res = client.get("/api/skills/saved")
    assert res.status_code == 200
    body = res.json()
    assert len(body) >= 1
    assert "id" in body[0]


def test_create_chat_defaults_to_ok_or_degraded():
    client = TestClient(app)
    res = client.post("/api/chat", json=_base_payload())
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in {"ok", "degraded"}
    assert "trace_id" in data


def test_use_saved_skill_requires_id():
    client = TestClient(app)
    payload = _base_payload()
    payload["features"] = ["rag", "use_saved_skill", "citation_guard"]
    payload["saved_skill_id"] = None
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 422


def test_web_search_failure_should_degrade_not_fail():
    client = TestClient(app)
    payload = _base_payload()
    payload["features"] = ["rag", "web_search", "citation_guard"]
    res = client.post("/api/chat", json=payload, headers={"x-fail-features": "web_search"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "degraded"
    assert "web_search" in data["degraded_features"]


def test_skill_exec_failure_should_degrade_not_fail():
    client = TestClient(app)
    payload = _base_payload()
    payload["features"] = ["rag", "skill_exec", "citation_guard"]
    res = client.post("/api/chat", json=payload, headers={"x-fail-features": "skill_exec"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "degraded"
    assert "skill_exec" in data["degraded_features"]


def test_generation_failure_should_fail_request():
    client = TestClient(app)
    res = client.post("/api/chat", json=_base_payload(), headers={"x-fail-features": "generation"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "failed"


def test_stream_done_event_contains_status_and_trace():
    client = TestClient(app)
    post_res = client.post("/api/chat", json=_base_payload(), headers={"x-fail-features": "web_search"})
    assert post_res.status_code == 200
    session_id = post_res.json()["session_id"]
    stream_res = client.get(f"/api/chat/stream?session_id={session_id}")
    assert stream_res.status_code == 200
    body = stream_res.text
    assert "event: done" in body
    assert "degraded_features" in body
    assert "trace_id" in body


def test_health_dependencies_endpoint():
    client = TestClient(app)
    res = client.get("/healthz/dependencies")
    assert res.status_code == 200
    data = res.json()
    assert data["app"] == "admissions-gateway"
    assert "dependencies" in data

