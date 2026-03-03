<script setup lang="ts">
import MessageBubble from './MessageBubble.vue'
import StreamBubble from './StreamBubble.vue'
import type { ChatMessage } from '../types'

const props = defineProps<{ messages: ChatMessage[]; streamingText: string; isStreaming: boolean }>()
</script>

<template>
  <section class="chat-main">
    <div class="status">
      <div class="status-title">系统状态</div>
      <div class="status-desc">数据来源已同步，建议开启联网搜索获取最新招生信息。</div>
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
  gap: 16px;
  overflow: hidden;
}

.status {
  padding: 12px 16px;
  border-radius: 14px;
  background: rgba(183, 139, 58, 0.12);
  border: 1px solid rgba(183, 139, 58, 0.25);
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

.messages {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow-y: auto;
  padding-right: 4px;
}
</style>
