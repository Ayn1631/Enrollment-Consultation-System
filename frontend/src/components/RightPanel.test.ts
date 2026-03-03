import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import RightPanel from './RightPanel.vue'
import type { HealthDependency } from '../types'

const deps: HealthDependency[] = [
  { name: 'retrieval-service', healthy: true, circuit_open: false, last_error: null },
  { name: 'web-search', healthy: false, circuit_open: false, last_error: 'timeout' }
]

function mountPanel(overrides: Record<string, unknown> = {}) {
  return mount(RightPanel, {
    props: {
      open: true,
      temperature: 0.6,
      topP: 0.9,
      model: 'zyit-gpt',
      strictCitation: true,
      healthLoading: false,
      reindexLoading: false,
      healthApp: 'admissions-gateway',
      healthOverall: false,
      dependencies: deps,
      reindexInfo: '',
      ...overrides
    }
  })
}

describe('RightPanel', () => {
  test('展示依赖状态并触发运维事件', async () => {
    const wrapper = mountPanel()

    expect(wrapper.text()).toContain('admissions-gateway')
    expect(wrapper.text()).toContain('retrieval-service')
    expect(wrapper.text()).toContain('web-search')

    const buttons = wrapper.findAll('.op-btn')
    expect(buttons.length).toBe(2)
    await buttons[0].trigger('click')
    await buttons[1].trigger('click')

    expect(wrapper.emitted('refreshHealth')).toBeTruthy()
    expect(wrapper.emitted('reindex')).toBeTruthy()
  })

  test('加载中时运维按钮禁用', () => {
    const wrapper = mountPanel({ healthLoading: true, reindexLoading: true })
    const buttons = wrapper.findAll('.op-btn')
    expect((buttons[0].element as HTMLButtonElement).disabled).toBe(true)
    expect((buttons[1].element as HTMLButtonElement).disabled).toBe(true)
  })
})
