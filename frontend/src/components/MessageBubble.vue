<script setup lang="ts">
import { computed } from 'vue'
import type { ChatMessage } from '../types'
import MarkdownContent from './MarkdownContent.vue'

const props = defineProps<{ message: ChatMessage }>()

const isAssistantMessage = computed(() => props.message.role === 'assistant')
const hasToolFeatures = computed(() => (props.message.enabledFeatures?.length ?? 0) > 0)
const shouldShowDegradedBanner = computed(
  () => hasToolFeatures.value && props.message.status === 'degraded' && Boolean(props.message.degradedFeatures?.length)
)
const hasExecutionDetails = computed(
  () =>
    hasToolFeatures.value &&
    isAssistantMessage.value &&
    Boolean(props.message.traceId || props.message.errorMessage || props.message.toolAudit?.length)
)
const shouldShowSources = computed(() => hasToolFeatures.value && Boolean(props.message.sources?.length))
const shouldShowAgentTrace = computed(
  () => hasToolFeatures.value && props.message.role === 'assistant' && Boolean(props.message.agentTrace?.length)
)
const detailSummary = computed(() => {
  if (props.message.status === 'failed') return '失败详情与追踪'
  if (props.message.status === 'degraded') return '降级详情与追踪'
  return '执行详情与追踪'
})
const detailCardClass = computed(() => {
  if (props.message.status === 'failed') return 'failure-card'
  if (props.message.status === 'degraded') return 'degraded-card'
  return 'detail-card'
})

const roleLabel = computed(() => {
  if (props.message.role === 'user') return '你'
  if (props.message.role === 'assistant') return '专家'
  return '系统'
})
</script>

<template>
  <div class="bubble" :class="props.message.role">
    <div class="meta">
      <span class="role">{{ roleLabel }}</span>
      <span class="time">{{ new Date(props.message.createdAt).toLocaleTimeString() }}</span>
    </div>
    <div v-if="shouldShowDegradedBanner" class="degraded-banner">
      部分能力降级：{{ props.message.degradedFeatures?.join(' / ') }}
    </div>
    <div v-if="props.message.status === 'failed'" class="degraded-banner failed">
      执行失败：{{ props.message.errorMessage || '本轮回答失败。' }}
    </div>
    <MarkdownContent class="content" :content="props.message.content" />

    <div v-if="shouldShowSources" class="sources">
      <div class="source" v-for="source in props.message.sources" :key="source.url">
        <span class="source-title">{{ source.title }}</span>
        <a :href="source.url" target="_blank" rel="noreferrer">查看来源</a>
      </div>
    </div>

    <details
      v-if="hasExecutionDetails"
      class="trace-card"
      :class="detailCardClass"
    >
      <summary>{{ detailSummary }}</summary>
      <div v-if="props.message.traceId" class="trace-row">
        <span class="trace-title">Trace</span>
        <span class="trace-message mono">{{ props.message.traceId }}</span>
      </div>
      <div v-if="props.message.errorMessage" class="trace-row">
        <span class="trace-title">错误</span>
        <span class="trace-message">{{ props.message.errorMessage }}</span>
      </div>
      <div v-if="props.message.toolAudit?.length" class="audit-list">
        <div class="trace-title">工具审计</div>
        <div class="audit-item" v-for="item in props.message.toolAudit" :key="item">
          <code>{{ item }}</code>
        </div>
      </div>
    </details>

    <details v-if="shouldShowAgentTrace" class="trace-card">
      <summary>专家轨迹</summary>
      <div class="trace-scroll">
        <div class="trace-row" v-for="item in props.message.agentTrace" :key="item.id">
          <span class="trace-title">{{ item.title }}</span>
          <span class="trace-status" :class="item.status">{{ item.status }}</span>
          <span v-if="item.subproblem_id" class="trace-meta">{{ item.subproblem_id }}</span>
          <span v-if="item.plan_step_index !== undefined" class="trace-meta">步骤 {{ item.plan_step_index }}</span>
          <span v-if="item.attempt !== undefined" class="trace-meta">尝试 {{ item.attempt }}</span>
          <span v-if="item.message" class="trace-message">{{ item.message }}</span>
        </div>
      </div>
    </details>
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

.trace-card {
  margin-top: 12px;
  border-radius: 12px;
  border: 1px solid rgba(27, 35, 32, 0.08);
  background: rgba(255, 255, 255, 0.72);
  padding: 10px 12px;
}

.trace-card summary {
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  color: var(--accent);
}

.failure-card {
  border-color: rgba(166, 30, 36, 0.16);
  background: rgba(166, 30, 36, 0.05);
}

.degraded-card {
  border-color: rgba(201, 133, 17, 0.2);
  background: rgba(201, 133, 17, 0.06);
}

.detail-card {
  border-color: rgba(31, 86, 166, 0.16);
  background: rgba(31, 86, 166, 0.04);
}

.trace-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
  font-size: 12px;
  color: var(--ink-1);
}

.trace-scroll {
  max-height: min(32vh, 280px);
  overflow-y: auto;
  padding-right: 4px;
}

.trace-title {
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

.audit-list {
  display: grid;
  gap: 6px;
  margin-top: 8px;
}

.audit-item code {
  display: block;
  padding: 8px 10px;
  border-radius: 8px;
  background: rgba(27, 35, 32, 0.06);
  color: var(--ink-1);
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}

.mono {
  font-family: ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace;
}
</style>
