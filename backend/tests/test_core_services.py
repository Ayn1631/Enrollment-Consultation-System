from __future__ import annotations

from datetime import datetime, timedelta

from app.contracts import MemoryEntry, RetrievalChunk
from app.services.memory import MemoryManager
from app.services.reranker import SimpleReranker
from app.services.skill_manager import SkillManager
from app.services.store import DocumentStore
from app.config import DOCS_DIR


def test_hybrid_retrieval_returns_scored_chunks():
    store = DocumentStore(DOCS_DIR)
    store.load()
    chunks = store.search("招生章程 学费 资助", top_k=5)
    assert len(chunks) > 0
    first = chunks[0]
    assert first.score > 0
    assert first.bm25_score >= 0
    assert first.vector_score >= 0
    assert first.keyword_score >= 0


def test_reranker_reorders_chunks():
    reranker = SimpleReranker()
    chunks = [
        RetrievalChunk(
            chunk_id="1",
            title="A",
            url="u1",
            text="这是招生章程内容",
            score=0.5,
            bm25_score=0.2,
            vector_score=0.1,
            keyword_score=0.0,
        ),
        RetrievalChunk(
            chunk_id="2",
            title="B",
            url="u2",
            text="这是学费和资助政策，含奖助贷",
            score=0.4,
            bm25_score=0.1,
            vector_score=0.2,
            keyword_score=0.2,
        ),
    ]
    ranked = reranker.rerank("学费资助政策", chunks, top_k=2)
    assert len(ranked) == 2
    assert ranked[0].chunk_id in {"1", "2"}


def test_memory_manager_supports_ttl():
    memory = MemoryManager()
    memory.write(
        "s1",
        MemoryEntry(
            key="k1",
            value="short memo",
            kind="short",
            expires_at=datetime.utcnow() - timedelta(seconds=1),
        ),
    )
    rows = memory.read("s1", kind="short")
    assert rows == []


def test_skill_manager_auto_version_and_active():
    manager = SkillManager()
    item1 = manager.save("tuition_helper", "步骤1->步骤2->给来源")
    item2 = manager.save("tuition_helper", "步骤1->步骤2->步骤3->给来源")
    assert item2.version >= item1.version
    active = manager.list_active()
    ids = {item.id for item in active}
    assert any(skill.startswith("tuition_helper") for skill in ids)
    note = manager.execute_saved(item2.id, "请解读学费")
    assert "已应用历史技能" in note

