<script setup lang="ts">
import type { ChatMessage } from '../types'

const props = defineProps<{ message: ChatMessage }>()
</script>

<template>
  <div class="bubble" :class="props.message.role">
    <div class="meta">
      <span class="role">{{ props.message.role === 'user' ? '你' : '系统' }}</span>
      <span class="time">{{ new Date(props.message.createdAt).toLocaleTimeString() }}</span>
    </div>
    <div v-if="props.message.status === 'degraded'" class="degraded-banner">
      部分能力降级：{{ props.message.degradedFeatures?.join(' / ') }}
    </div>
    <div v-if="props.message.status === 'failed'" class="degraded-banner failed">生成服务异常，本轮回答失败。</div>
    <p class="content">{{ props.message.content }}</p>

    <div v-if="props.message.sources?.length" class="sources">
      <div class="source" v-for="source in props.message.sources" :key="source.url">
        <span class="source-title">{{ source.title }}</span>
        <a :href="source.url" target="_blank" rel="noreferrer">查看来源</a>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bubble {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid rgba(27, 35, 32, 0.08);
  animation: bubble-in 0.4s ease both;
}

.bubble.user {
  align-self: flex-end;
  background: rgba(166, 30, 36, 0.12);
  border-color: rgba(166, 30, 36, 0.28);
}

.bubble.assistant {
  align-self: flex-start;
  background: rgba(198, 40, 50, 0.08);
  border-color: rgba(198, 40, 50, 0.24);
}

.bubble.system {
  align-self: center;
  background: rgba(127, 21, 27, 0.1);
  border-color: rgba(127, 21, 27, 0.24);
}

.meta {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: var(--ink-2);
  margin-bottom: 8px;
}

.content {
  margin: 0;
  white-space: pre-wrap;
  line-height: 1.6;
}

.degraded-banner {
  margin-bottom: 8px;
  padding: 6px 8px;
  border-radius: 8px;
  background: rgba(166, 30, 36, 0.14);
  color: var(--accent-cool);
  font-size: 12px;
}

.degraded-banner.failed {
  background: rgba(198, 57, 57, 0.2);
  color: #7f1e1e;
}

.sources {
  margin-top: 12px;
  display: grid;
  gap: 8px;
}

.source {
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(27, 35, 32, 0.08);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
}

.source-title {
  color: var(--ink-1);
}

a {
  color: var(--accent);
  text-decoration: none;
}
</style>
