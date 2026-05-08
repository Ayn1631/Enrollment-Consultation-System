import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, test, vi } from 'vitest'
import App from './App.vue'

describe('App', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        if (url.includes('/healthz/dependencies')) {
          return {
            ok: true,
            json: async () => ({
              app: 'admissions-gateway',
              healthy: true,
              dependencies: [{ name: 'rag-agent-service', healthy: true, circuit_open: false, last_error: null }]
            })
          } as Response
        }
        throw new Error(`Unexpected fetch: ${url}`)
      })
    )
  })

  test('渲染新的单页工作台骨架', async () => {
    const wrapper = mount(App, {
      global: {
        stubs: {
          MarkdownContent: {
            props: ['content'],
            template: '<div class="markdown-stub">{{ content }}</div>'
          }
        }
      }
    })

    await wrapper.vm.$nextTick()

    expect(wrapper.text()).toContain('中原工学院招生咨询系统')
    expect(wrapper.text()).toContain('模式选择')
    expect(wrapper.text()).toContain('模型调优')
    expect(wrapper.text()).toContain('Prompt Seeds')
  })

  test('切换到专家模式时显示固定轨迹区域', async () => {
    const wrapper = mount(App, {
      global: {
        stubs: {
          MarkdownContent: {
            props: ['content'],
            template: '<div class="markdown-stub">{{ content }}</div>'
          }
        }
      }
    })

    const expertButton = wrapper.findAll('button').find((item) => item.text().includes('专家模式'))
    expect(expertButton).toBeTruthy()
    await expertButton!.trigger('click')

    expect(wrapper.text()).toContain('最近一轮专家轨迹')
    expect(wrapper.text()).toContain('专家模式的轨迹会显示在这里')
  })
})
