import { startChatStream } from '../services/api'
import type { AgentStepEvent, ChatRequest, ChatStreamEvent } from '../types'

export function useStream() {
  const startStream = async (
    request: ChatRequest,
    handlers: {
      onDelta: (delta: string) => void
      onStep: (event: AgentStepEvent) => void
      onDone: (event: ChatStreamEvent) => void
      onError: (err: Error) => void
    }
  ): Promise<() => void> => {
    return startChatStream(request, handlers)
  }

  return { startStream }
}
