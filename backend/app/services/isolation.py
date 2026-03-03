from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class IsolationResult(Generic[T]):
    ok: bool
    value: T | None = None
    error: str | None = None
    degraded: bool = False


@dataclass(slots=True)
class CircuitState:
    open_until: float = 0.0
    failures: int = 0
    last_error: str | None = None
    updated_at: datetime = datetime.utcnow()


class IsolationExecutor:
    def __init__(self, failure_threshold: int = 2, open_seconds: float = 5.0):
        # 关键变量：failure_threshold 控制连续失败多少次后打开熔断。
        self.failure_threshold = failure_threshold
        # 关键变量：open_seconds 控制熔断保持时间，避免雪崩重试。
        self.open_seconds = open_seconds
        self._states: dict[str, CircuitState] = {}
        self._lock = Lock()

    def execute(self, name: str, func: Callable[[], T]) -> IsolationResult[T]:
        """执行受保护调用，内置失败计数、熔断和自动恢复。"""
        with self._lock:
            state = self._states.setdefault(name, CircuitState())
            now = time.time()
            if state.open_until > now:
                return IsolationResult(ok=False, error=f"circuit_open:{name}", degraded=True)

        try:
            value = func()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                state = self._states.setdefault(name, CircuitState())
                state.failures += 1
                state.last_error = str(exc)
                state.updated_at = datetime.utcnow()
                if state.failures >= self.failure_threshold:
                    state.open_until = time.time() + self.open_seconds
            return IsolationResult(ok=False, error=str(exc), degraded=True)

        with self._lock:
            state = self._states.setdefault(name, CircuitState())
            state.failures = 0
            state.last_error = None
            state.open_until = 0.0
            state.updated_at = datetime.utcnow()
        return IsolationResult(ok=True, value=value)

    def snapshot(self) -> dict[str, CircuitState]:
        """返回当前隔离状态快照，用于健康检查和运维面板展示。"""
        with self._lock:
            return dict(self._states)
