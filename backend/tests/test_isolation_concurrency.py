from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from app.main import app


def _payload() -> dict:
    return {
        "session_id": uuid.uuid4().hex,
        "messages": [{"role": "user", "content": "请解读招生政策与学费"}],
        "mode": "chat",
        "stream": True,
        "features": ["rag", "web_search", "skill_exec", "citation_guard"],
        "strict_citation": True,
    }


def _send(client: TestClient, fail_features: str = "") -> dict:
    headers = {"x-fail-features": fail_features} if fail_features else {}
    res = client.post("/api/chat", json=_payload(), headers=headers)
    assert res.status_code == 200
    return res.json()


def test_parallel_requests_keep_available_when_partial_failures():
    client = TestClient(app)

    def worker(idx: int) -> dict:
        if idx % 3 == 0:
            return _send(client, "web_search")
        if idx % 5 == 0:
            return _send(client, "skill_exec")
        return _send(client)

    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(worker, range(20)))

    assert len(rows) == 20
    degraded_count = sum(1 for item in rows if item["status"] == "degraded")
    ok_count = sum(1 for item in rows if item["status"] in {"ok", "degraded"})
    assert degraded_count >= 1
    assert ok_count == 20

