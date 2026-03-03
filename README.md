# Enrollment-Consultation-System

## 项目说明

本项目实现了招生咨询系统的可用 MVP，核心能力包括：

- 前端功能可选开关：`rag`、`web_search`、`skill_exec`、`use_saved_skill`、`citation_guard`
- 历史技能下拉单选（启用 `use_saved_skill` 时生效）
- 后端网关编排 + 功能降级（除生成服务外，其他功能故障自动降级）
- 服务级拆分：`retrieval`、`rerank`、`memory`、`skill`、`generation`、`observability`
- RAG/Agent 技术栈：`LangChain` + `LangGraph` + `Neo4j` + `LangChain4j Bridge`
- SSE 流式返回，`done` 事件携带 `status/degraded_features/sources/trace_id`
- 接口健康检查与依赖状态查看

## 目录

- `frontend/` Vue3 + Vite
- `backend/` FastAPI 网关与服务层（含 `service_apps` 微服务）
- `docs/` 招生资料源数据
- `docker-compose.yml` 一体化启动（前端/后端/Postgres/Redis）

## 后端运行

```bash
cd backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

默认会读取 `backend/.env`。

关键环境变量：
- `SERVICE_CALL_MODE=local|http`：`local` 为进程内调用；`http` 通过各微服务 URL 调用。
- `ADMIN_API_TOKEN`：为空时管理接口免鉴权；非空时调用 `/api/admin/*` 必须带 `x-admin-token`。
- `RETRIEVAL_SERVICE_URL`、`RERANK_SERVICE_URL`、`MEMORY_SERVICE_URL`、`SKILL_SERVICE_URL`、`GENERATION_SERVICE_URL`：仅 `http` 模式使用。
- `RAG_STACK=langchain|native`：`langchain` 优先走 LangChain 检索并可叠加 Neo4j 事实。
- `AGENT_STACK=langgraph|native`：`langgraph` 使用 LangGraph 规划功能执行顺序。
- `NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD/NEO4J_DATABASE`：启用 Neo4j 图谱增强所需。
- `LANGCHAIN4J_SERVICE_URL`：配置后，历史技能优先通过 LangChain4j 服务执行。

## 前端运行

```bash
cd frontend
npm install
npm run dev
```

默认 `VITE_USE_MOCK=true`，若要联调后端请设置为 `false` 并配置 `VITE_API_BASE_URL`。
若后端启用 `ADMIN_API_TOKEN`，前端需配置 `VITE_ADMIN_API_TOKEN` 以调用重建索引接口。

## 测试

后端：

```bash
cd backend
pytest -q
```

前端：

```bash
cd frontend
npm run test
npm run build
```

CI 会自动执行：
- 后端：`pytest` + `evaluate_gateway.py` + `gate_release.py`
- 前端：`vitest` + `vite build`

## Docker Compose

```bash
docker compose up --build
```

## 评测与发布门禁

```bash
cd backend
python scripts/evaluate_gateway.py
python scripts/gate_release.py
```

会在 `backend/reports/eval_report.json` 生成评测报告，并根据门槛给出发布是否通过。

可选发布门槛配置：
- `GATE_MIN_PASS_RATE`（默认 `0.8`）
- `GATE_MAX_P95_MS`（默认 `1500`）
- `GATE_MAX_FAILED_ROWS`（默认 `2`）
- `GATE_REPORT_PATH`（默认 `reports/eval_report.json`）
