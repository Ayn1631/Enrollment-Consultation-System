import { startChatStream } from '../services/api'
import type { ChatRequest, ChatStreamEvent } from '../types'

export function useStream() {
  const useMock = import.meta.env.VITE_USE_MOCK === 'true'

  const startStream = async (
    request: ChatRequest,
    handlers: {
      onDelta: (delta: string) => void
      onDone: (event: ChatStreamEvent) => void
      onError: (err: Error) => void
    }
  ): Promise<() => void> => {
    if (useMock) {
      const { startMockStream } = await import('../services/mockApi')
      return startMockStream(request, {
        onDelta: handlers.onDelta,
        onDone: handlers.onDone
      })
    }

    return startChatStream(request, handlers)
  }

  return { startStream }
}
