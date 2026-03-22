import type {
  ChatRequest,
  ChatStreamEvent,
  FeatureMeta,
  HealthResponse,
  SavedSkill
} from '../types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
const ADMIN_TOKEN = import.meta.env.VITE_ADMIN_API_TOKEN ?? ''

export class ApiRequestError extends Error {
  status?: number
  traceId?: string

  constructor(message: string, options?: { status?: number; traceId?: string }) {
    super(message)
    this.name = 'ApiRequestError'
    this.status = options?.status
    this.traceId = options?.traceId
  }
}

function adminHeaders(): Record<string, string> {
  if (!ADMIN_TOKEN) {
    return {}
  }
  return { 'x-admin-token': ADMIN_TOKEN }
}

async function readJsonSafely(res: Response): Promise<Record<string, unknown> | null> {
  try {
    const payload = (await res.json()) as unknown
    if (payload && typeof payload === 'object') {
      return payload as Record<string, unknown>
    }
  } catch {
    return null
  }
  return null
}

async function ensureOk(res: Response, fallbackMessage: string): Promise<void> {
  if (res.ok) return
  const payload = await readJsonSafely(res)
  const detail = payload?.detail
  const detailObject = detail && typeof detail === 'object' ? (detail as Record<string, unknown>) : null
  const traceId =
    (typeof payload?.trace_id === 'string' ? payload.trace_id : undefined) ??
    (typeof detailObject?.trace_id === 'string' ? detailObject.trace_id : undefined)
  const errorMessage =
    (typeof detail === 'string' ? detail : undefined) ??
    (typeof detailObject?.message === 'string' ? detailObject.message : undefined) ??
    (typeof payload?.message === 'string' ? payload.message : undefined) ??
    `${fallbackMessage}（HTTP ${res.status}）`
  const finalMessage = traceId ? `${errorMessage}（trace_id: ${traceId}）` : errorMessage
  throw new ApiRequestError(finalMessage, { status: res.status, traceId })
}

export async function postChat(request: ChatRequest): Promise<{
  session_id: string
  trace_id?: string
  status?: 'ok' | 'degraded' | 'failed'
  degraded_features?: string[]
}> {
  console.log('[api.postChat] request', {
    apiBase: API_BASE,
    session_id: request.session_id,
    features: request.features,
    mode: request.mode,
    strict_citation: request.strict_citation,
    model: request.model,
    message_count: request.messages.length
  })
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(request)
  })

  await ensureOk(res, '提交聊天请求失败')
  const payload = await res.json()
  console.log('[api.postChat] response', payload)
  return payload
}

export function openChatStream(
  sessionId: string,
  handlers: {
    onDelta: (delta: string) => void
    onDone: (event: ChatStreamEvent) => void
    onError: (err: Error) => void
  }
): () => void {
  const url = `${API_BASE}/api/chat/stream?session_id=${encodeURIComponent(sessionId)}`
  console.log('[api.openChatStream] connect', { url, sessionId })
  const source = new EventSource(url)

  const handleMessage = (ev: MessageEvent) => {
    try {
      const data = JSON.parse(ev.data) as ChatStreamEvent
      console.log('[api.openChatStream] message', data)
      if (data.delta) {
        handlers.onDelta(data.delta)
      }
    } catch {
      console.log('[api.openChatStream] message(raw)', ev.data)
      handlers.onDelta(String(ev.data))
    }
  }

  const handleDone = (ev: MessageEvent) => {
    try {
      const data = JSON.parse(ev.data) as ChatStreamEvent
      console.log('[api.openChatStream] done', data)
      source.close()
      handlers.onDone(data)
    } catch {
      console.log('[api.openChatStream] done(raw)', ev.data)
      source.close()
      handlers.onDone({ finish_reason: 'stop' })
    }
  }

  const handleError = () => {
    console.error('[api.openChatStream] error', { url, sessionId })
    source.close()
    handlers.onError(new ApiRequestError('与后端流式连接中断，请检查后端服务或接口地址配置。'))
  }

  source.addEventListener('message', handleMessage)
  source.addEventListener('done', handleDone)
  source.addEventListener('error', handleError)

  return () => source.close()
}

export async function getTools(): Promise<Array<{ id: string; label: string }>> {
  const res = await fetch(`${API_BASE}/api/tools`)
  await ensureOk(res, '获取工具列表失败')
  return res.json()
}

export async function getFeatures(): Promise<FeatureMeta[]> {
  const res = await fetch(`${API_BASE}/api/features`)
  await ensureOk(res, '获取功能列表失败')
  return res.json()
}

export async function getSavedSkills(): Promise<SavedSkill[]> {
  const res = await fetch(`${API_BASE}/api/skills/saved`)
  await ensureOk(res, '获取历史技能失败')
  return res.json()
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/healthz/dependencies`)
  await ensureOk(res, '获取后端健康状态失败')
  return res.json()
}

export async function postReindex(): Promise<{ status: string; result: { chunks: number } }> {
  const res = await fetch(`${API_BASE}/api/admin/reindex`, {
    method: 'POST',
    headers: adminHeaders()
  })
  await ensureOk(res, '重建索引失败')
  return res.json()
}
