import { openChatStream, postChat } from '../services/api'
import { startMockStream } from '../services/mockApi'
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
      return startMockStream(request, {
        onDelta: handlers.onDelta,
        onDone: () => handlers.onDone({ finish_reason: 'stop' })
      })
    }

    const res = await postChat(request)
    const sessionId = res.session_id || request.session_id
    return openChatStream(sessionId, handlers)
  }

  return { startStream }
}
