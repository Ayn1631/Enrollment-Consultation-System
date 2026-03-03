import type { ChatRequest, ChatStreamEvent } from '../types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export async function postChat(request: ChatRequest): Promise<{ session_id: string }> {
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
