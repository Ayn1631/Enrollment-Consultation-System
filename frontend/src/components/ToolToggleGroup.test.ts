import { mount } from '@vue/test-utils'
import { describe, expect, test } from 'vitest'
import ToolToggleGroup from './ToolToggleGroup.vue'
import type { FeatureMeta } from '../types'

const options: FeatureMeta[] = [
  { id: 'rag', label: '本地RAG检索', default_enabled: true, dependencies: [] },
  { id: 'use_saved_skill', label: '使用以往技能', default_enabled: false, dependencies: ['skill_exec'] }
]

describe('ToolToggleGroup', () => {
  test('支持多选开关并触发更新', async () => {
    const wrapper = mount(ToolToggleGroup, {
      props: {
        modelValue: ['rag'],
        options
      }
    })

    await wrapper.get('.dropdown-trigger').trigger('click')
    const buttons = wrapper.findAll('.dropdown-item')
    expect(buttons.length).toBe(2)

    await buttons[1].trigger('click')
    const emitted = wrapper.emitted('update:modelValue')
    expect(emitted).toBeTruthy()
    const lastPayload = emitted?.[emitted.length - 1]?.[0] as string[]
    expect(lastPayload).toEqual(['rag', 'use_saved_skill'])
  })
})
