import { buildChatRequest, validateFeatureSelection } from './requestBuilder'
import { describe, expect, test } from 'vitest'

describe('requestBuilder', () => {
  test('默认保留 features 并自动开启 strict citation', () => {
    const req = buildChatRequest({
      sessionId: 's1',
      messages: [{ role: 'user', content: '你好' }],
      features: ['rag', 'citation_guard'],
      mode: 'chat',
      stream: true,
      savedSkillId: '',
      strictCitation: false,
      temperature: 0.4,
      topP: 0.9,
      model: 'zyit-gpt'
    })
    expect(req.features).toEqual(['rag', 'citation_guard'])
    expect(req.strict_citation).toBe(true)
  })

  test('开启 use_saved_skill 未选择技能时返回错误', () => {
    const err = validateFeatureSelection(['rag', 'use_saved_skill'], '')
    expect(err).toContain('请先选择一个历史技能')
  })

  test('同时启用 rag + web_search + skill_exec 时请求字段完整', () => {
    const req = buildChatRequest({
      sessionId: 's2',
      messages: [{ role: 'user', content: '招生政策' }],
      features: ['rag', 'web_search', 'skill_exec'],
      mode: 'chat',
      stream: true,
      savedSkillId: '',
      strictCitation: false,
      temperature: 0.2,
      topP: 0.8,
      model: 'zyit-pro'
    })
    expect(req.features).toEqual(['rag', 'web_search', 'skill_exec'])
    expect(req.temperature).toBe(0.2)
    expect(req.top_p).toBe(0.8)
    expect(req.model).toBe('zyit-pro')
  })
})
