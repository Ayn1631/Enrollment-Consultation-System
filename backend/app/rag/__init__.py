"""LangChain + LangGraph RAG runtime."""

from __future__ import annotations

from typing import Any

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # noqa: BLE001
    _tqdm = None


class _NoopProgress:
    def __enter__(self) -> "_NoopProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def update(self, _: int = 1) -> None:
        return None

    def set_postfix(
        self,
        ordered_dict: dict[str, Any] | None = None,
        refresh: bool = True,
        **kwargs: Any,
    ) -> None:
        return None

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        return None

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        return None

    def set_description_str(self, desc: str | None = None, refresh: bool = True) -> None:
        return None

    def close(self) -> None:
        return None


def build_progress(*, enabled: bool, **kwargs: Any):
    """按需创建 tqdm；缺依赖或禁用时回退为空对象。"""
    if not enabled or _tqdm is None:
        return _NoopProgress()
    options = {
        "dynamic_ncols": True,
        "mininterval": 0.2,
    }
    options.update(kwargs)
    return _tqdm(**options)
