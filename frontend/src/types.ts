export type Role = 'user' | 'assistant' | 'system'
export type FeatureFlag = 'rag' | 'web_search' | 'skill_exec' | 'use_saved_skill' | 'citation_guard'
export type ToolMode = 'search' | 'react' | 'plan' | 'guide' // legacy compatibility
export type ChatMode = 'chat' | 'rag' | 'plan' | 'guide' | 'agent'
export type ChatStatus = 'ok' | 'degraded' | 'failed'
export type AgentStrategy = 'speed' | 'quality'
export type AgentStepStatus = 'started' | 'completed' | 'retrying' | 'degraded' | 'failed'

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

export interface HealthDependency {
  name: string
  healthy: boolean
  circuit_open: boolean
  last_error?: string | null
}

export interface HealthResponse {
  app: string
  healthy: boolean
  dependencies: HealthDependency[]
}

export interface MemoryCompressionResponse {
  session_id: string
  long_summary: string
  long_memory_count: number
  special_memory_count: number
  route: 'light' | 'main' | 'requested' | 'mock'
  model: string
  notes: string[]
}

export interface AgentStepEvent {
  id: string
  node: string
  title: string
  status: AgentStepStatus
  message?: string
  subproblem_id?: string
  plan_step_index?: number
  attempt?: number
  strategy: AgentStrategy
  timestamp: string
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
  errorMessage?: string
  toolAudit?: string[]
  agentTrace?: AgentStepEvent[]
}

export interface ChatSession {
  id: string
  sessionId: string
  title: string
  createdAt: string
  updatedAt: string
  messages: ChatMessage[]
  streamingText: string
  isStreaming: boolean
  latestDegradedFeatures: FeatureFlag[]
  currentAgentTrace: AgentStepEvent[]
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
  agent_strategy?: AgentStrategy
}

export interface ChatStreamEvent {
  delta?: string
  id?: string
  node?: string
  title?: string
  message?: string
  subproblem_id?: string
  plan_step_index?: number
  attempt?: number
  strategy?: AgentStrategy
  timestamp?: string
  agent_strategy?: AgentStrategy
  status?: ChatStatus
  degraded_features?: FeatureFlag[]
  trace_id?: string
  sources?: ChatSource[]
  tool_audit?: string[]
  error_message?: string
  finish_reason?: 'stop' | 'length' | 'error'
}
