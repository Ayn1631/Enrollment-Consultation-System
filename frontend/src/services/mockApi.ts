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
  _request: ChatRequest,
  handlers: {
    onDelta: (delta: string) => void
    onDone: (event: { finish_reason: 'stop' }) => void
  }
): () => void {
  const tokens = demoText.split('')
  let index = 0

  const timer = setInterval(() => {
    if (index >= tokens.length) {
      clearInterval(timer)
      handlers.onDone({ finish_reason: 'stop' })
      return
    }
    handlers.onDelta(tokens[index])
    index += 1
  }, 24)

  return () => {
    clearInterval(timer)
    handlers.onDone({ finish_reason: 'stop' })
  }
}
