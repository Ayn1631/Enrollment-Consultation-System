from __future__ import annotations

from fastapi.testclient import TestClient

from app.service_apps.generation_api import app as generation_app
from app.service_apps.memory_api import app as memory_app
from app.service_apps.rerank_api import app as rerank_app
from app.service_apps.retrieval_api import app as retrieval_app
from app.service_apps.skill_api import app as skill_app


def test_retrieval_service_endpoint():
    client = TestClient(retrieval_app)
    res = client.post("/retrieve", json={"query": "招生章程", "top_k": 3})
    assert res.status_code == 200
    assert "chunks" in res.json()


def test_rerank_service_endpoint():
    client = TestClient(rerank_app)
    payload = {
        "query": "学费资助",
        "chunks": [
            {
                "chunk_id": "c1",
                "title": "t1",
                "url": "u1",
                "text": "学费政策内容",
                "score": 0.2,
                "bm25_score": 0.2,
                "vector_score": 0.1,
                "keyword_score": 0.0,
            }
        ],
        "top_k": 1,
    }
    res = client.post("/rerank", json=payload)
    assert res.status_code == 200
    assert len(res.json()["chunks"]) == 1


def test_memory_service_write_read():
    client = TestClient(memory_app)
    write_res = client.post(
        "/memory/write",
        json={
            "session_id": "s-memory",
            "entry": {"key": "k", "value": "v", "kind": "short", "confidence": 0.9, "source": "test"},
        },
    )
    assert write_res.status_code == 200
    read_res = client.post("/memory/read?kind=short", json={"session_id": "s-memory"})
    assert read_res.status_code == 200
    assert len(read_res.json()["entries"]) >= 1


def test_skill_service_saved_list_and_execute():
    client = TestClient(skill_app)
    list_res = client.get("/skills/saved?active_only=true")
    assert list_res.status_code == 200
    skills = list_res.json()["skills"]
    assert len(skills) >= 1
    first = skills[0]
    exec_res = client.post(
        "/skills/execute",
        json={"query": "招生流程", "session_id": "s1", "saved_skill_id": first["id"]},
    )
    assert exec_res.status_code == 200
    assert "已应用历史技能" in exec_res.json()["note"]


def test_generation_service_endpoint():
    client = TestClient(generation_app)
    res = client.post(
        "/generate",
        json={
            "user_query": "请介绍招生政策",
            "context_blocks": ["章程第一条", "章程第二条"],
            "feature_notes": ["RAG 已执行"],
        },
    )
    assert res.status_code == 200
    assert "text" in res.json()

