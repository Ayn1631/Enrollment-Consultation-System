import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import StreamBubble from './StreamBubble.vue'

function mountBubble(mode: 'chat' | 'rag' | 'agent') {
  return mount(StreamBubble, {
    props: {
      content: '',
      waitingFirstChunk: true,
      mode
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

describe('StreamBubble', () => {
  test('对话模式不显示检索文案', () => {
    const wrapper = mountBubble('chat')
    expect(wrapper.text()).toContain('正在生成回答，请稍等...')
    expect(wrapper.text()).not.toContain('正在检索资料并生成回答')
  })

  test('RAG 模式显示检索文案', () => {
    const wrapper = mountBubble('rag')
    expect(wrapper.text()).toContain('正在检索资料并生成回答，请稍等...')
  })
})
