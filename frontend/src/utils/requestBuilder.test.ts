import { buildChatRequest, resolveModeFeatures } from './requestBuilder'
import { describe, expect, test } from 'vitest'

describe('requestBuilder', () => {
  test('对话模式默认纯对话，不启用 RAG', () => {
    const req = buildChatRequest({
      sessionId: 's1',
      messages: [{ role: 'user', content: '你好' }],
      mode: 'chat',
      stream: true,
      temperature: 0.4,
      topP: 0.9,
      model: 'gpt-5.4',
      agentStrategy: 'speed'
    })
    expect(req.features).toEqual([])
    expect(req.strict_citation).toBe(false)
    expect(req.agent_strategy).toBe('speed')
  })

  test('专家模式自动启用全功能链路', () => {
    const req = buildChatRequest({
      sessionId: 's2',
      messages: [{ role: 'user', content: '招生政策' }],
      mode: 'agent',
      stream: true,
      temperature: 0.2,
      topP: 0.8,
      model: 'zyit-pro',
      agentStrategy: 'quality'
    })
    expect(req.features).toEqual(['rag', 'web_search', 'skill_exec', 'citation_guard'])
    expect(req.temperature).toBe(0.2)
    expect(req.top_p).toBe(0.8)
    expect(req.model).toBe('zyit-pro')
    expect(req.agent_strategy).toBe('quality')
  })

  test('RAG 问答模式固定启用 RAG 与引用校验', () => {
    expect(resolveModeFeatures('rag')).toEqual(['rag', 'citation_guard'])
  })
})
