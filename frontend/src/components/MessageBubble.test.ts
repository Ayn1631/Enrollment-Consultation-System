import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import MessageBubble from './MessageBubble.vue'

function mountBubble(overrides: Record<string, unknown> = {}) {
  return mount(MessageBubble, {
    props: {
      message: {
        id: 'msg-1',
        role: 'assistant',
        content: '测试内容',
        createdAt: '2026-03-26T00:00:00.000Z',
        traceId: 'trace-001',
        toolAudit: ['generation:requested:gpt-5.4:cache_miss'],
        ...overrides
      }
    },
    global: {
      stubs: {
        MarkdownContent: {
          props: ['content'],
          template: '<div class="markdown-stub">{{ content }}</div>'
        }
      }
    }
  })
}

describe('MessageBubble', () => {
  test('成功消息显示执行详情而不是失败详情', () => {
    const wrapper = mountBubble({ status: 'ok', enabledFeatures: ['rag'] })

    expect(wrapper.text()).toContain('执行详情与追踪')
    expect(wrapper.text()).not.toContain('失败详情与追踪')
    expect(wrapper.find('.detail-card').exists()).toBe(true)
  })

  test('失败消息显示失败详情', () => {
    const wrapper = mountBubble({ status: 'failed', errorMessage: 'Request timed out.', enabledFeatures: ['rag'] })

    expect(wrapper.text()).toContain('失败详情与追踪')
    expect(wrapper.find('.failure-card').exists()).toBe(true)
  })

  test('降级消息显示降级详情', () => {
    const wrapper = mountBubble({ status: 'degraded', errorMessage: 'Request timed out.', enabledFeatures: ['rag'] })

    expect(wrapper.text()).toContain('降级详情与追踪')
    expect(wrapper.find('.degraded-card').exists()).toBe(true)
  })

  test('对话模式消息不显示来源和执行详情', () => {
    const wrapper = mountBubble({
      status: 'ok',
      enabledFeatures: [],
      sources: [{ title: '招生章程', url: 'https://example.com' }],
      toolAudit: ['agent_tool:local_rag_search'],
      traceId: 'trace-chat-001',
      agentTrace: [
        {
          id: 'step-1',
          node: 'load_memory',
          title: '加载记忆',
          status: 'completed',
          strategy: 'speed',
          timestamp: '2026-03-27T00:00:00.000Z'
        }
      ]
    })

    expect(wrapper.text()).not.toContain('查看来源')
    expect(wrapper.text()).not.toContain('执行详情与追踪')
    expect(wrapper.text()).not.toContain('专家轨迹')
  })
})
