import type { ChatMode, ChatRequest, FeatureFlag, Role } from '../types'

export interface RequestBuilderInput {
  sessionId: string
  messages: Array<{ role: Role; content: string }>
  features: FeatureFlag[]
  mode: ChatMode
  stream: boolean
  savedSkillId: string
  strictCitation: boolean
  temperature: number
  topP: number
  model: string
}

export function validateFeatureSelection(features: FeatureFlag[], savedSkillId: string): string | null {
  if (features.includes('use_saved_skill') && !savedSkillId) {
    return '已开启“使用以往技能”，请先选择一个历史技能。'
  }
  return null
}

export function buildChatRequest(input: RequestBuilderInput): ChatRequest {
  const strictCitation = input.strictCitation || input.features.includes('citation_guard')
  return {
    session_id: input.sessionId,
    messages: input.messages,
    features: input.features,
    mode: input.mode,
    stream: input.stream,
    saved_skill_id: input.features.includes('use_saved_skill') ? input.savedSkillId : undefined,
    strict_citation: strictCitation,
    temperature: input.temperature,
    top_p: input.topP,
    model: input.model
  }
}

