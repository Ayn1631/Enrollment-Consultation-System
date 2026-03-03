import type {
  ChatRequest,
  ChatStreamEvent,
  FeatureMeta,
  HealthResponse,
  SavedSkill
} from '../types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export async function postChat(request: ChatRequest): Promise<{
  session_id: string
  trace_id?: string
  status?: 'ok' | 'degraded' | 'failed'
  degraded_features?: string[]
}> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(request)
  })

  if (!res.ok) {
    throw new Error(`POST /api/chat failed: ${res.status}`)
  }

  return res.json()
}

export function openChatStream(
  sessionId: string,
  handlers: {
    onDelta: (delta: string) => void
    onDone: (event: ChatStreamEvent) => void
    onError: (err: Event) => void
  }
): () => void {
  const url = `${API_BASE}/api/chat/stream?session_id=${encodeURIComponent(sessionId)}`
  const source = new EventSource(url)

  const handleMessage = (ev: MessageEvent) => {
    try {
      const data = JSON.parse(ev.data) as ChatStreamEvent
      if (data.delta) {
        handlers.onDelta(data.delta)
      }
    } catch {
      handlers.onDelta(String(ev.data))
    }
  }

  const handleDone = (ev: MessageEvent) => {
    try {
      const data = JSON.parse(ev.data) as ChatStreamEvent
      handlers.onDone(data)
    } catch {
      handlers.onDone({ finish_reason: 'stop' })
    }
  }

  source.addEventListener('message', handleMessage)
  source.addEventListener('done', handleDone)
  source.addEventListener('error', handlers.onError)

  return () => source.close()
}

export async function getTools(): Promise<Array<{ id: string; label: string }>> {
  const res = await fetch(`${API_BASE}/api/tools`)
  if (!res.ok) {
    throw new Error(`GET /api/tools failed: ${res.status}`)
  }
  return res.json()
}

export async function getFeatures(): Promise<FeatureMeta[]> {
  const res = await fetch(`${API_BASE}/api/features`)
  if (!res.ok) {
    throw new Error(`GET /api/features failed: ${res.status}`)
  }
  return res.json()
}

export async function getSavedSkills(): Promise<SavedSkill[]> {
  const res = await fetch(`${API_BASE}/api/skills/saved`)
  if (!res.ok) {
    throw new Error(`GET /api/skills/saved failed: ${res.status}`)
  }
  return res.json()
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/healthz/dependencies`)
  if (!res.ok) {
    throw new Error(`GET /healthz/dependencies failed: ${res.status}`)
  }
  return res.json()
}

export async function postReindex(): Promise<{ status: string; result: { chunks: number } }> {
  const res = await fetch(`${API_BASE}/api/admin/reindex`, { method: 'POST' })
  if (!res.ok) {
    throw new Error(`POST /api/admin/reindex failed: ${res.status}`)
  }
  return res.json()
}
