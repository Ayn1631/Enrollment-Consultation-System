<script setup lang="ts">
import MessageBubble from './MessageBubble.vue'
import StreamBubble from './StreamBubble.vue'
import type { ChatMessage, FeatureFlag } from '../types'

const props = defineProps<{
  messages: ChatMessage[]
  streamingText: string
  isStreaming: boolean
  activeFeatures: FeatureFlag[]
  degradedFeatures: FeatureFlag[]
}>()
</script>

<template>
  <section class="chat-main">
    <div class="status">
      <div class="status-title">系统状态</div>
      <div class="status-desc">已启用：{{ props.activeFeatures.join(' / ') || '无' }}</div>
      <div v-if="props.degradedFeatures.length" class="status-degraded">
        已降级：{{ props.degradedFeatures.join(' / ') }}
      </div>
    </div>

    <div class="messages">
      <MessageBubble v-for="message in props.messages" :key="message.id" :message="message" />
      <StreamBubble v-if="props.isStreaming" :content="props.streamingText" />
    </div>
  </section>
</template>

<style scoped>
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 12px;
  overflow: hidden;
  min-height: 0;
}

.status {
  padding: 10px 14px;
  border-radius: 14px;
  background: rgba(166, 30, 36, 0.1);
  border: 1px solid rgba(166, 30, 36, 0.2);
  font-size: 13px;
}

.status-title {
  font-weight: 600;
  margin-bottom: 4px;
  color: var(--accent);
}

.status-desc {
  color: var(--ink-1);
}

.status-degraded {
  margin-top: 6px;
  color: var(--accent-cool);
  font-weight: 600;
}

.messages {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 12px;
  overflow-y: auto;
  padding-right: 4px;
  min-height: 0;
}
</style>
