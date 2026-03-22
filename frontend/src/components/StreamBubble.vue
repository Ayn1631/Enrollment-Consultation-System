<script setup lang="ts">
import MarkdownContent from './MarkdownContent.vue'

const props = defineProps<{ content: string }>()
</script>

<template>
  <div class="bubble assistant streaming">
    <div class="meta">
      <span class="role">系统</span>
      <span class="time">流式输出中</span>
    </div>
    <div class="content">
      <MarkdownContent :content="props.content" />
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

.caret {
  display: inline-block;
  margin-left: 4px;
  animation: blink 0.9s step-start infinite;
}
</style>
