<script setup lang="ts">
import type { ChatMode } from '../types'

const props = defineProps<{ modelValue: ChatMode }>()
const emit = defineEmits<{ (e: 'update:modelValue', value: ChatMode): void }>()

const options: Array<{ value: ChatMode; label: string }> = [
  { value: 'chat', label: '对话模式' },
  { value: 'plan', label: '规划执行' },
  { value: 'guide', label: '指引模式' }
]
</script>

<template>
  <div class="mode-list">
    <button
      v-for="option in options"
      :key="option.value"
      class="mode"
      :class="{ active: option.value === props.modelValue }"
      @click="emit('update:modelValue', option.value)"
    >
      {{ option.label }}
    </button>
  </div>
</template>

<style scoped>
.mode-list {
  display: grid;
  gap: 8px;
}

.mode {
  padding: 10px 14px;
  border-radius: 999px;
  border: 1px solid var(--line-soft);
  background: rgba(255, 255, 255, 0.7);
  color: var(--ink-1);
  cursor: pointer;
  transition: 0.2s ease;
}

.mode.active {
  color: var(--accent);
  border-color: rgba(183, 139, 58, 0.6);
  background: rgba(183, 139, 58, 0.16);
}
</style>
