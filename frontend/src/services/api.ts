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

function parseSseBlock(block: string): { event: string; data: string } | null {
  const lines = block
    .split('\n')
    .map((line) => line.trimEnd())
    .filter(Boolean)
  if (!lines.length) return null
  let event = 'message'
  const dataLines: string[] = []
  for (const line of lines) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim() || 'message'
      continue
    }
    if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trimStart())
    }
  }
  if (!dataLines.length) return null
  return { event, data: dataLines.join('\n') }
}

export async function startChatStream(
  request: ChatRequest,
  handlers: {
    onDelta: (delta: string) => void
    onDone: (event: ChatStreamEvent) => void
    onError: (err: Error) => void
  }
): Promise<() => void> {
  const controller = new AbortController()
  console.log('[api.startChatStream] request', {
    apiBase: API_BASE,
    session_id: request.session_id,
    features: request.features,
    mode: request.mode,
    strict_citation: request.strict_citation,
    model: request.model,
    message_count: request.messages.length
  })
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: {
      Accept: 'text/event-stream',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(request),
    signal: controller.signal
  })
  await ensureOk(res, '发起流式聊天请求失败')
  if (!res.body) {
    throw new ApiRequestError('后端未返回可读取的流式响应体。')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let closed = false
  let doneReceived = false

  const cleanup = () => {
    if (closed) return
    closed = true
    controller.abort()
    void reader.cancel().catch(() => undefined)
  }

  const dispatchBlock = (block: string) => {
    const parsed = parseSseBlock(block)
    if (!parsed) return
    if (parsed.event === 'done') {
      const data = JSON.parse(parsed.data) as ChatStreamEvent
      console.log('[api.startChatStream] done', data)
      doneReceived = true
      cleanup()
      handlers.onDone(data)
      return
    }
    try {
      const data = JSON.parse(parsed.data) as ChatStreamEvent
      console.log('[api.startChatStream] message', data)
      if (data.delta) {
        handlers.onDelta(data.delta)
      }
    } catch {
      console.log('[api.startChatStream] message(raw)', parsed.data)
      handlers.onDelta(parsed.data)
    }
  }

  void (async () => {
    try {
      while (!closed) {
        const { done, value } = await reader.read()
        if (done) {
          break
        }
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
        let splitIndex = buffer.indexOf('\n\n')
        while (splitIndex !== -1) {
          const block = buffer.slice(0, splitIndex)
          buffer = buffer.slice(splitIndex + 2)
          dispatchBlock(block)
          splitIndex = buffer.indexOf('\n\n')
        }
      }
      buffer += decoder.decode().replace(/\r\n/g, '\n')
      if (buffer.trim()) {
        dispatchBlock(buffer.trim())
      }
      if (!doneReceived && !closed) {
        cleanup()
        handlers.onError(new ApiRequestError('流式连接已结束，但后端没有返回 done 事件。'))
      }
    } catch (error) {
      if (closed || controller.signal.aborted) {
        return
      }
      console.error('[api.startChatStream] error', error)
      cleanup()
      handlers.onError(
        error instanceof Error
          ? error
          : new ApiRequestError('与后端流式连接中断，请检查后端服务或接口地址配置。')
      )
    }
  })()

  return () => {
    console.log('[api.startChatStream] cancel', { sessionId: request.session_id })
    cleanup()
  }
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
