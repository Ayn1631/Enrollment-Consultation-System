import type { ChatRequest } from '../types'

const demoText = `中原工学院欢迎你！这里是招生咨询系统的演示流式输出。

你可以：
- 使用“RAG问答模式”基于校内资料进行问答
- 使用“专家模式”获取更完整的工具增强答案

示例问题：
1) 2025年招生章程有哪些关键时间点？
2) 学费与资助政策有哪些类型？
3) 新郑龙湖校区的生活服务有哪些？`

export function startMockStream(
  request: ChatRequest,
  handlers: {
    onDelta: (delta: string) => void
    onStep?: (event: {
      id: string
      node: string
      title: string
      status: 'started' | 'completed' | 'retrying' | 'degraded' | 'failed'
      strategy: 'speed' | 'quality'
      timestamp: string
    }) => void
    onDone: (event: {
      finish_reason: 'stop'
      status?: 'ok' | 'degraded'
      degraded_features?: Array<'rag' | 'web_search' | 'skill_exec' | 'use_saved_skill' | 'citation_guard'>
      sources?: Array<{ title: string; url: string }>
      trace_id?: string
    }) => void
  }
): () => void {
  handlers.onStep?.({
    id: `mock-step-${Date.now()}`,
    node: 'mock_agent',
    title: '演示步骤',
    status: 'completed',
    strategy: request.agent_strategy ?? 'speed',
    timestamp: new Date().toISOString()
  })
  const tokens = demoText.split('')
  let index = 0

  const timer = setInterval(() => {
    if (index >= tokens.length) {
      clearInterval(timer)
      handlers.onDone({
        finish_reason: 'stop',
        status: request.features.includes('web_search') ? 'degraded' : 'ok',
        degraded_features: request.features.includes('web_search') ? ['web_search'] : [],
        sources: [
          {
            title: '中原工学院2025年普通本科招生章程',
            url: 'https://zsc.zut.edu.cn/info/1124/2673.htm'
          }
        ],
        trace_id: `mock-${Date.now()}`
      })
      return
    }
    handlers.onDelta(tokens[index])
    index += 1
  }, 24)

  return () => {
    clearInterval(timer)
    handlers.onDone({ finish_reason: 'stop', status: 'ok', degraded_features: [] })
  }
}
