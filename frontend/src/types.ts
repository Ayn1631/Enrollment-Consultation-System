export type Role = 'user' | 'assistant' | 'system'
export type ToolMode = 'search' | 'react' | 'plan' | 'guide'
export type ChatMode = 'chat' | 'plan' | 'guide'

export interface ChatSource {
  title: string
  url: string
}

export interface ChatMessage {
  id: string
  role: Role
  content: string
  createdAt: string
  sources?: ChatSource[]
}

export interface ChatRequest {
  session_id: string
  messages: Array<{ role: Role; content: string }>
  tools: ToolMode[]
  mode: ChatMode
  stream: boolean
}

export interface ChatStreamEvent {
  delta?: string
  finish_reason?: 'stop' | 'length' | 'error'
}
