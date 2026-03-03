from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from app.models import SessionResult
from app.services.isolation import IsolationExecutor


@dataclass
class SessionStore:
    _sessions: dict[str, SessionResult] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def set(self, session_id: str, value: SessionResult) -> None:
        with self._lock:
            self._sessions[session_id] = value

    def get(self, session_id: str) -> SessionResult | None:
        with self._lock:
            return self._sessions.get(session_id)


@dataclass
class ServiceContainer:
    session_store: SessionStore = field(default_factory=SessionStore)
    isolation: IsolationExecutor = field(default_factory=IsolationExecutor)

