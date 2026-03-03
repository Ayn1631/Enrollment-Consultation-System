export type Role = 'user' | 'assistant' | 'system'
export type FeatureFlag = 'rag' | 'web_search' | 'skill_exec' | 'use_saved_skill' | 'citation_guard'
export type ToolMode = 'search' | 'react' | 'plan' | 'guide' // legacy compatibility
export type ChatMode = 'chat' | 'plan' | 'guide'
export type ChatStatus = 'ok' | 'degraded' | 'failed'

export interface ChatSource {
  title: string
  url: string
}

export interface FeatureMeta {
  id: FeatureFlag
  label: string
  default_enabled: boolean
  dependencies: FeatureFlag[]
}

export interface SavedSkill {
  id: string
  label: string
  description: string
}

export interface ChatMessage {
  id: string
  role: Role
  content: string
  createdAt: string
  status?: ChatStatus
  degradedFeatures?: FeatureFlag[]
  enabledFeatures?: FeatureFlag[]
  traceId?: string
  sources?: ChatSource[]
}

export interface ChatRequest {
  session_id: string
  messages: Array<{ role: Role; content: string }>
  features: FeatureFlag[]
  tools?: ToolMode[]
  mode: ChatMode
  stream: boolean
  saved_skill_id?: string
  strict_citation?: boolean
  temperature?: number
  top_p?: number
  model?: string
}

export interface ChatStreamEvent {
  delta?: string
  status?: ChatStatus
  degraded_features?: FeatureFlag[]
  trace_id?: string
  sources?: ChatSource[]
  finish_reason?: 'stop' | 'length' | 'error'
}
