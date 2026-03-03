from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.contracts import (
    SkillExecuteRequest,
    SkillExecuteResponse,
    SkillListResponse,
    SkillSaveRequest,
)
from app.services.skill_manager import SkillManager


app = FastAPI(title="skill-service")
skills = SkillManager()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "skill-service"}


@app.post("/skills/execute", response_model=SkillExecuteResponse)
def execute_skill(request: SkillExecuteRequest) -> SkillExecuteResponse:
    if request.saved_skill_id:
        try:
            note = skills.execute_saved(request.saved_skill_id, request.query)
        except ValueError as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SkillExecuteResponse(note=note)
    note = skills.execute_general(request.query)
    return SkillExecuteResponse(note=note)


@app.post("/skills/save")
def save_skill(request: SkillSaveRequest):
    item = skills.save(request.name, request.workflow)
    return item


@app.get("/skills/saved", response_model=SkillListResponse)
def list_saved_skills(active_only: bool = True) -> SkillListResponse:
    if active_only:
        return SkillListResponse(skills=skills.list_active())
    return SkillListResponse(skills=skills.list_versions())

