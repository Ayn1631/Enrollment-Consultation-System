from __future__ import annotations

from app.config import DOCS_DIR, Settings
from app.services.service_client import ServiceClient
from app.services.store import DocumentStore


def _local_settings() -> Settings:
    return Settings(
        service_call_mode="local",
        use_mock_generation=True,
        docs_dir=DOCS_DIR,
        api_key="",
        api_url="https://example.com/v1/chat/completions",
    )


def test_dependency_health_local_mode():
    settings = _local_settings()
    store = DocumentStore(settings.docs_dir)
    store.load()
    client = ServiceClient(settings=settings, store=store)
    health = client.dependency_health()
    assert health["retrieval-service"]["healthy"] is True
    assert health["generation-service"]["healthy"] is True


def test_service_client_skill_save_and_list():
    settings = _local_settings()
    store = DocumentStore(settings.docs_dir)
    store.load()
    client = ServiceClient(settings=settings, store=store)
    saved = client.save_skill("custom_flow", "步骤A->步骤B->给来源")
    assert saved["name"] == "custom_flow"
    active = client.list_saved_skills()
    assert len(active.skills) >= 1
