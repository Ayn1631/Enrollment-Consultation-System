import type { AgentStrategy, ChatMode, ChatRequest, FeatureFlag, Role } from '../types'

export interface RequestBuilderInput {
  sessionId: string
  messages: Array<{ role: Role; content: string }>
  mode: ChatMode
  stream: boolean
  temperature: number
  topP: number
  model: string
  agentStrategy: AgentStrategy
}

export function resolveModeFeatures(mode: ChatMode): FeatureFlag[] {
  if (mode === 'research' || mode === 'agent') {
    return ['rag', 'web_search', 'skill_exec', 'citation_guard']
  }
  if (mode === 'rag' || mode === 'expert') {
    return ['rag', 'citation_guard']
  }
  return []
}

export function buildChatRequest(input: RequestBuilderInput): ChatRequest {
  const features = resolveModeFeatures(input.mode)
  const normalizedMode = input.mode === 'agent' ? 'research' : input.mode
  const strictCitation = normalizedMode === 'rag' || normalizedMode === 'expert' || normalizedMode === 'research'
  return {
    session_id: input.sessionId,
    messages: input.messages,
    features,
    mode: normalizedMode,
    stream: input.stream,
    saved_skill_id: undefined,
    strict_citation: strictCitation,
    temperature: input.temperature,
    top_p: input.topP,
    model: input.model,
    agent_strategy: input.agentStrategy
  }
}
