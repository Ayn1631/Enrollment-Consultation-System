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
      mode: 'agent',
      temperature: 0.6,
      topP: 0.9,
      model: 'gpt-5.4',
      agentStrategy: 'speed',
      healthLoading: false,
      reindexLoading: false,
      compressLoading: false,
      healthApp: 'admissions-gateway',
      healthOverall: false,
      dependencies: deps,
      reindexInfo: '',
      compressInfo: '',
      canCompressContext: true,
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
    expect(buttons.length).toBe(3)
    await buttons[0].trigger('click')
    await buttons[1].trigger('click')
    await buttons[2].trigger('click')

    expect(wrapper.emitted('refreshHealth')).toBeTruthy()
    expect(wrapper.emitted('reindex')).toBeTruthy()
    expect(wrapper.emitted('compressContext')).toBeTruthy()
  })

  test('加载中时运维按钮禁用', () => {
    const wrapper = mountPanel({ healthLoading: true, reindexLoading: true, compressLoading: true })
    const buttons = wrapper.findAll('.op-btn')
    expect((buttons[0].element as HTMLButtonElement).disabled).toBe(true)
    expect((buttons[1].element as HTMLButtonElement).disabled).toBe(true)
    expect((buttons[2].element as HTMLButtonElement).disabled).toBe(true)
  })

  test('可以切换执行策略', async () => {
    const wrapper = mountPanel()
    const buttons = wrapper.findAll('.strategy-btn')
    expect(buttons.length).toBe(2)
    await buttons[1].trigger('click')
    expect(wrapper.emitted('update:agentStrategy')?.[0]).toEqual(['quality'])
  })

  test('非专家模式下隐藏执行策略', () => {
    const wrapper = mountPanel({ mode: 'chat' })
    expect(wrapper.findAll('.strategy-btn')).toHaveLength(0)
  })
})
