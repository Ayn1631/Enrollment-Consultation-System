<script setup lang="ts">
import ToolToggleGroup from './ToolToggleGroup.vue'
import ModeSelector from './ModeSelector.vue'
import type { ChatMode, ToolMode } from '../types'

const props = defineProps<{ tools: ToolMode[]; mode: ChatMode }>()
const emit = defineEmits<{
  (e: 'update:tools', value: ToolMode[]): void
  (e: 'update:mode', value: ChatMode): void
}>()
</script>

<template>
  <aside class="sidebar">
    <div class="panel">
      <div class="panel-title">功能权限</div>
      <ToolToggleGroup :model-value="props.tools" @update:model-value="emit('update:tools', $event)" />
    </div>

    <div class="panel">
      <div class="panel-title">对话方式</div>
      <ModeSelector :model-value="props.mode" @update:model-value="emit('update:mode', $event)" />
    </div>

    <div class="panel sessions">
      <div class="panel-title">最近会话</div>
      <div class="session">招生章程解读</div>
      <div class="session">学费与资助</div>
      <div class="session">新生报到流程</div>
      <button class="new-session">新建会话</button>
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
  gap: 14px;
}

.panel {
  padding: 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid var(--line-soft);
  box-shadow: var(--shadow-soft);
}

.panel-title {
  font-size: 12px;
  color: var(--ink-2);
  margin-bottom: 12px;
  text-transform: uppercase;
  letter-spacing: 0.18em;
}

.sessions .session {
  padding: 10px 12px;
  border-radius: 12px;
  margin-bottom: 8px;
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid transparent;
  transition: 0.2s ease;
}

.sessions .session:hover {
  border-color: rgba(183, 139, 58, 0.4);
  background: rgba(183, 139, 58, 0.12);
}

.new-session {
  width: 100%;
  padding: 10px;
  border-radius: 12px;
  border: 1px dashed rgba(183, 139, 58, 0.5);
  background: transparent;
  color: var(--accent);
  cursor: pointer;
}

.notice p {
  margin: 0 0 8px;
  color: var(--ink-1);
  line-height: 1.5;
}
</style>
