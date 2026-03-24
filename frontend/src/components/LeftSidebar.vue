<script setup lang="ts">
import ToolToggleGroup from './ToolToggleGroup.vue'
import ModeSelector from './ModeSelector.vue'
import type { ChatMode, ChatSession, FeatureFlag, FeatureMeta, SavedSkill } from '../types'

const props = defineProps<{
  features: FeatureFlag[]
  featureOptions: FeatureMeta[]
  mode: ChatMode
  savedSkills: SavedSkill[]
  savedSkillId: string
  sessions: ChatSession[]
  activeSessionId: string
}>()
const emit = defineEmits<{
  (e: 'update:features', value: FeatureFlag[]): void
  (e: 'update:mode', value: ChatMode): void
  (e: 'update:savedSkillId', value: string): void
  (e: 'switchSession', sessionId: string): void
  (e: 'createSession'): void
  (e: 'clearSession'): void
}>()

const formatSessionTime = (value: string) =>
  new Date(value).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit'
  })

const getSessionSummary = (session: ChatSession) => {
  const lastMessage =
    [...session.messages].reverse().find((item) => item.role !== 'system') ??
    session.messages[session.messages.length - 1]
  if (!lastMessage) {
    return '暂无消息'
  }
  return lastMessage.content.trim().replace(/\s+/g, ' ').slice(0, 28) || '暂无消息'
}
</script>

<template>
  <aside class="sidebar">
    <div class="panel">
      <div class="panel-title">功能权限</div>
      <ToolToggleGroup
        :model-value="props.features"
        :options="props.featureOptions"
        @update:model-value="emit('update:features', $event)"
      />
      <div v-if="props.features.includes('use_saved_skill')" class="saved-skill">
        <label class="skill-label" for="savedSkill">历史技能</label>
        <select
          id="savedSkill"
          :value="props.savedSkillId"
          @change="emit('update:savedSkillId', ($event.target as HTMLSelectElement).value)"
        >
          <option value="">请选择技能</option>
          <option v-for="skill in props.savedSkills" :key="skill.id" :value="skill.id">
            {{ skill.label }}
          </option>
        </select>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">对话方式</div>
      <ModeSelector :model-value="props.mode" @update:model-value="emit('update:mode', $event)" />
    </div>

    <div class="panel sessions">
      <div class="panel-head">
        <div class="panel-title">最近会话</div>
        <button class="panel-link" type="button" @click="emit('clearSession')">清空当前</button>
      </div>
      <button
        v-for="session in props.sessions"
        :key="session.id"
        type="button"
        class="session"
        :class="{ active: session.id === props.activeSessionId }"
        @click="emit('switchSession', session.id)"
      >
        <div class="session-top">
          <span class="session-title">{{ session.title }}</span>
          <span class="session-time">{{ formatSessionTime(session.updatedAt) }}</span>
        </div>
        <div class="session-summary">
          {{ session.isStreaming ? '正在生成回答...' : getSessionSummary(session) }}
        </div>
      </button>
      <button class="new-session" type="button" @click="emit('createSession')">新建会话</button>
    </div>

    <div class="panel notice">
      <div class="panel-title">招生快讯</div>
      <p>· 2025年招生章程已发布</p>
      <p>· 校园开放日预约通道已上线</p>
      <p>· 重要时间节点请关注招生官网</p>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
  overflow-y: auto;
  padding-right: 4px;
}

.panel {
  padding: 14px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid var(--line-soft);
  box-shadow: var(--shadow-soft);
}

.panel-title {
  font-size: 12px;
  color: var(--ink-2);
  margin-bottom: 10px;
  text-transform: uppercase;
  letter-spacing: 0.18em;
}

.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 10px;
}

.panel-head .panel-title {
  margin-bottom: 0;
}

.panel-link {
  border: none;
  background: transparent;
  color: var(--accent);
  font-size: 12px;
  cursor: pointer;
  padding: 0;
}

.sessions .session {
  width: 100%;
  padding: 9px 11px;
  border-radius: 12px;
  margin-bottom: 6px;
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(166, 30, 36, 0.08);
  transition: 0.2s ease;
  text-align: left;
  cursor: pointer;
}

.sessions .session:hover {
  border-color: rgba(166, 30, 36, 0.34);
  background: rgba(166, 30, 36, 0.1);
}

.sessions .session.active {
  background: rgba(166, 30, 36, 0.14);
  border-color: rgba(166, 30, 36, 0.38);
}

.session-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.session-title {
  color: var(--ink-0);
  font-size: 13px;
  font-weight: 600;
}

.session-time,
.session-summary {
  color: var(--ink-2);
  font-size: 12px;
}

.session-summary {
  margin-top: 6px;
  line-height: 1.45;
}

.new-session {
  width: 100%;
  padding: 9px;
  border-radius: 12px;
  border: 1px dashed rgba(166, 30, 36, 0.45);
  background: transparent;
  color: var(--accent);
  cursor: pointer;
}

.notice p {
  margin: 0 0 6px;
  color: var(--ink-1);
  line-height: 1.45;
}

.saved-skill {
  margin-top: 10px;
  display: grid;
  gap: 6px;
}

.saved-skill select {
  border: 1px solid var(--line-soft);
  border-radius: 10px;
  padding: 8px 10px;
  background: rgba(255, 255, 255, 0.95);
}

.skill-label {
  font-size: 12px;
  color: var(--ink-2);
}
</style>
