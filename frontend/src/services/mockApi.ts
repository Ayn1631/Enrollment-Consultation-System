import type { ChatRequest } from '../types'

const demoText = `中原工学院欢迎你！这里是招生咨询系统的演示流式输出。

你可以：
- 开启“联网搜索”获取最新招生动态
- 使用“规划执行”让系统分步骤完成任务
- 使用“指引模式”进行流程化咨询

示例问题：
1) 2025年招生章程有哪些关键时间点？
2) 学费与资助政策有哪些类型？
3) 新郑龙湖校区的生活服务有哪些？`

export function startMockStream(
  request: ChatRequest,
  handlers: {
    onDelta: (delta: string) => void
    onDone: (event: {
      finish_reason: 'stop'
      status?: 'ok' | 'degraded'
      degraded_features?: Array<'rag' | 'web_search' | 'skill_exec' | 'use_saved_skill' | 'citation_guard'>
      sources?: Array<{ title: string; url: string }>
      trace_id?: string
    }) => void
  }
): () => void {
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
