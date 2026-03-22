<script setup lang="ts">
import { computed } from 'vue'
import { renderMarkdown } from '../utils/markdown'

const props = defineProps<{ content: string }>()
const renderedContent = computed(() => renderMarkdown(props.content))
</script>

<template>
  <div class="bubble assistant streaming">
    <div class="meta">
      <span class="role">系统</span>
      <span class="time">流式输出中</span>
    </div>
    <div class="content markdown-body">
      <div v-html="renderedContent"></div>
      <span class="caret">▍</span>
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

.content :deep(p) {
  margin: 0;
}

.content :deep(p + p),
.content :deep(p + ul),
.content :deep(ul + p),
.content :deep(pre + p),
.content :deep(p + pre) {
  margin-top: 10px;
}

.content :deep(ul) {
  margin: 0;
  padding-left: 20px;
}

.content :deep(code) {
  font-family: 'Consolas', 'SFMono-Regular', monospace;
  font-size: 0.92em;
  padding: 0.14em 0.38em;
  border-radius: 6px;
  background: rgba(127, 21, 27, 0.08);
}

.content :deep(pre) {
  margin: 0;
  padding: 12px 14px;
  border-radius: 12px;
  background: rgba(80, 14, 18, 0.92);
  color: #fff8f7;
  overflow-x: auto;
}

.content :deep(pre code) {
  padding: 0;
  background: transparent;
  color: inherit;
}

.content :deep(a) {
  color: var(--accent);
  text-decoration: none;
}

.caret {
  display: inline-block;
  margin-left: 4px;
  animation: blink 0.9s step-start infinite;
}
</style>
