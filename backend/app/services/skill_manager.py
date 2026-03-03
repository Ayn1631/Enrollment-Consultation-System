from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

from app.contracts import SkillVersionItem


@dataclass(slots=True)
class SkillVersion:
    id: str
    name: str
    version: int
    workflow_hash: str
    score: float
    active: bool
    created_at: datetime
    description: str
    workflow: str


class SkillManager:
    def __init__(self):
        self._skills: dict[str, list[SkillVersion]] = {}
        self._lock = Lock()
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        self.save("admission_faq", "提取问题关键词->检索政策条款->给结论与来源")
        self.save("fee_and_aid", "识别费用类型->匹配学费和资助条款->输出费用明细")
        self.save("new_student_guide", "识别新生阶段->输出报到/住宿/医保流程")

    def _evaluate(self, workflow: str) -> float:
        score = 0.45
        if "->" in workflow:
            score += 0.2
        if len(workflow) > 20:
            score += 0.2
        if "来源" in workflow or "条款" in workflow:
            score += 0.1
        return min(score, 0.99)

    def save(self, name: str, workflow: str) -> SkillVersionItem:
        workflow_hash = hashlib.sha256(workflow.encode("utf-8")).hexdigest()[:16]
        score = self._evaluate(workflow)
        with self._lock:
            versions = self._skills.setdefault(name, [])
            for existing in versions:
                if existing.workflow_hash == workflow_hash:
                    return self._to_item(existing)
            next_version = len(versions) + 1
            skill_id = f"{name}_v{next_version}"
            new_version = SkillVersion(
                id=skill_id,
                name=name,
                version=next_version,
                workflow_hash=workflow_hash,
                score=score,
                active=False,
                created_at=datetime.utcnow(),
                description=f"自动生成技能版本 {next_version}",
                workflow=workflow,
            )
            versions.append(new_version)
            self._activate_best(name)
            return self._to_item(new_version)

    def _activate_best(self, name: str) -> None:
        versions = self._skills[name]
        best = max(versions, key=lambda x: (x.score, x.version))
        for item in versions:
            item.active = item.id == best.id

    def list_versions(self) -> list[SkillVersionItem]:
        with self._lock:
            rows: list[SkillVersionItem] = []
            for versions in self._skills.values():
                rows.extend(self._to_item(v) for v in versions)
            rows.sort(key=lambda x: (x.name, x.version), reverse=False)
            return rows

    def list_active(self) -> list[SkillVersionItem]:
        with self._lock:
            rows = [
                self._to_item(v)
                for versions in self._skills.values()
                for v in versions
                if v.active
            ]
            rows.sort(key=lambda x: x.name)
            return rows

    def execute_saved(self, skill_id: str, query: str) -> str:
        with self._lock:
            for versions in self._skills.values():
                for version in versions:
                    if version.id == skill_id:
                        return (
                            f"已应用历史技能 {version.id}（score={version.score:.2f}）："
                            f"针对“{query[:32]}”执行 {version.workflow}"
                        )
        raise ValueError("saved skill not found")

    def execute_general(self, query: str) -> str:
        active = self.list_active()
        if not active:
            return f"未找到激活技能，已按通用链路处理 query={query[:30]}"
        best = active[0]
        return f"已执行技能 {best.id}：query={query[:30]}"

    def _to_item(self, version: SkillVersion) -> SkillVersionItem:
        return SkillVersionItem(
            id=version.id,
            name=version.name,
            version=version.version,
            workflow_hash=version.workflow_hash,
            score=version.score,
            active=version.active,
            created_at=version.created_at,
            description=version.description,
        )

