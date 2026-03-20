import { openChatStream, postChat } from '../services/api'
import type { ChatRequest, ChatStreamEvent } from '../types'

export function useStream() {
  const useMock = import.meta.env.VITE_USE_MOCK !== 'false'

  const startStream = async (
    request: ChatRequest,
    handlers: {
      onDelta: (delta: string) => void
      onDone: (event: ChatStreamEvent) => void
      onError: (err: Event) => void
    }
  ): Promise<() => void> => {
    if (useMock) {
      const { startMockStream } = await import('../services/mockApi')
      return startMockStream(request, {
        onDelta: handlers.onDelta,
        onDone: handlers.onDone
      })
    }

    const res = await postChat(request)
    const sessionId = res.session_id || request.session_id
    return openChatStream(sessionId, handlers)
  }

  return { startStream }
}
