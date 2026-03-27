<script setup lang="ts">
import { computed } from 'vue'
import MarkdownContent from './MarkdownContent.vue'
import type { ChatMode } from '../types'

const props = defineProps<{
  content: string
  waitingFirstChunk: boolean
  mode: ChatMode
}>()

const loadingText = computed(() => {
  if (props.mode === 'chat') {
    return '正在生成回答，请稍等...'
  }
  if (props.mode === 'rag') {
    return '正在检索资料并生成回答，请稍等...'
  }
  return '正在调用专家能力并生成回答，请稍等...'
})
</script>

<template>
  <div class="bubble assistant streaming">
    <div class="meta">
      <span class="role">系统</span>
      <span class="time">{{ props.waitingFirstChunk ? '正在准备回答' : '流式输出中' }}</span>
    </div>
    <div class="content">
      <div v-if="props.waitingFirstChunk" class="loading-state" aria-live="polite">
        <span class="spinner" aria-hidden="true"></span>
        <span class="loading-text">{{ loadingText }}</span>
      </div>
      <template v-else>
        <MarkdownContent :content="props.content" />
        <span class="caret">▍</span>
      </template>
    </div>
  </div>
</template>

<style scoped>
.bubble {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(166, 30, 36, 0.1);
  border: 1px solid rgba(166, 30, 36, 0.26);
  animation: bubble-in 0.4s ease both;
}

.meta {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: var(--ink-2);
  margin-bottom: 8px;
}

.content {
  line-height: 1.6;
}

.loading-state {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  min-height: 28px;
  color: var(--ink-1);
}

.spinner {
  width: 16px;
  height: 16px;
  border-radius: 999px;
  border: 2px solid rgba(166, 30, 36, 0.18);
  border-top-color: var(--accent);
  animation: spin 0.8s linear infinite;
  flex-shrink: 0;
}

.loading-text {
  font-size: 13px;
}

.caret {
  display: inline-block;
  margin-left: 4px;
  animation: blink 0.9s step-start infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }

  to {
    transform: rotate(360deg);
  }
}
</style>
