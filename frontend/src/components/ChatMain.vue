<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import MessageBubble from './MessageBubble.vue'
import StreamBubble from './StreamBubble.vue'
import type { AgentStepEvent, ChatMessage, ChatMode, FeatureFlag } from '../types'

const props = defineProps<{
  messages: ChatMessage[]
  streamingText: string
  isStreaming: boolean
  mode: ChatMode
  degradedFeatures: FeatureFlag[]
  currentAgentTrace: AgentStepEvent[]
}>()

const messagesRef = ref<HTMLElement | null>(null)

const scrollToBottom = async () => {
  await nextTick()
  const container = messagesRef.value
  if (!container) return
  container.scrollTop = container.scrollHeight
}

watch(
  () => [props.messages.length, props.streamingText, props.isStreaming],
  () => {
    void scrollToBottom()
  },
  { flush: 'post' }
)
</script>

<template>
  <section class="chat-main">
    <div v-if="props.mode !== 'chat' && props.degradedFeatures.length" class="status">
      <div class="status-title">能力降级提醒</div>
      <div class="status-degraded">已降级：{{ props.degradedFeatures.join(' / ') }}</div>
    </div>

    <div v-if="props.mode === 'agent' && props.currentAgentTrace.length" class="trace-panel">
      <div class="trace-title">当前执行轨迹</div>
      <div class="trace-list">
        <div class="trace-item" v-for="item in props.currentAgentTrace" :key="item.id">
          <span class="trace-node">{{ item.title }}</span>
          <span class="trace-status" :class="item.status">{{ item.status }}</span>
          <span v-if="item.subproblem_id" class="trace-meta">{{ item.subproblem_id }}</span>
          <span v-if="item.plan_step_index !== undefined" class="trace-meta">步骤 {{ item.plan_step_index }}</span>
          <span v-if="item.attempt !== undefined" class="trace-meta">尝试 {{ item.attempt }}</span>
          <span v-if="item.message" class="trace-message">{{ item.message }}</span>
        </div>
      </div>
    </div>

    <div ref="messagesRef" class="messages">
      <MessageBubble v-for="message in props.messages" :key="message.id" :message="message" />
      <StreamBubble
        v-if="props.isStreaming"
        :content="props.streamingText"
        :waiting-first-chunk="props.streamingText.length === 0"
        :mode="props.mode"
      />
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

.status-degraded {
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

.trace-panel {
  border: 1px solid rgba(27, 35, 32, 0.08);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.76);
  padding: 10px 12px;
  display: grid;
  gap: 8px;
  max-height: min(36vh, 320px);
  min-height: 0;
}

.trace-list {
  display: grid;
  gap: 8px;
  overflow-y: auto;
  min-height: 0;
  padding-right: 4px;
}

.trace-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--accent);
}

.trace-item {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 12px;
  color: var(--ink-1);
}

.trace-node {
  font-weight: 600;
}

.trace-status {
  border-radius: 999px;
  padding: 0 8px;
  background: rgba(27, 35, 32, 0.08);
}

.trace-status.completed {
  background: rgba(55, 155, 92, 0.14);
  color: #1e6d3f;
}

.trace-status.retrying,
.trace-status.degraded,
.trace-status.failed {
  background: rgba(166, 30, 36, 0.12);
  color: var(--accent);
}

.trace-meta,
.trace-message {
  color: var(--ink-2);
}

.trace-message {
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
