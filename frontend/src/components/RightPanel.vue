<script setup lang="ts">
import type { HealthDependency } from '../types'

const props = defineProps<{
  open: boolean
  temperature: number
  topP: number
  model: string
  strictCitation: boolean
  healthLoading: boolean
  reindexLoading: boolean
  healthApp: string
  healthOverall: boolean
  dependencies: HealthDependency[]
  reindexInfo: string
}>()

const emit = defineEmits<{
  (e: 'toggle'): void
  (e: 'update:temperature', value: number): void
  (e: 'update:topP', value: number): void
  (e: 'update:model', value: string): void
  (e: 'update:strictCitation', value: boolean): void
  (e: 'refreshHealth'): void
  (e: 'reindex'): void
}>()
</script>

<template>
  <aside class="right" :class="{ closed: !props.open }">
    <button class="toggle" @click="emit('toggle')">
      {{ props.open ? '收起' : '展开' }}
    </button>

    <div v-if="props.open" class="content">
      <div class="panel-title">系统参数</div>

      <div class="control">
        <label>温度 Temperature</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          :value="props.temperature"
          @input="emit('update:temperature', Number(($event.target as HTMLInputElement).value))"
        />
        <span>{{ props.temperature.toFixed(2) }}</span>
      </div>

      <div class="control">
        <label>Top_p</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          :value="props.topP"
          @input="emit('update:topP', Number(($event.target as HTMLInputElement).value))"
        />
        <span>{{ props.topP.toFixed(2) }}</span>
      </div>

      <div class="control">
        <label>模型选择</label>
        <select :value="props.model" @change="emit('update:model', ($event.target as HTMLSelectElement).value)">
          <option value="zyit-gpt">Zyit-GPT</option>
          <option value="zyit-edu">Zyit-EDU</option>
          <option value="zyit-pro">Zyit-Pro</option>
        </select>
      </div>

      <div class="control toggle-control">
        <label>严格引用模式</label>
        <input
          type="checkbox"
          :checked="props.strictCitation"
          @change="emit('update:strictCitation', ($event.target as HTMLInputElement).checked)"
        />
      </div>

      <div class="note">提示：后端对接完成后，这些参数将实时影响模型响应。</div>

      <div class="panel-title">运行状态</div>
      <div class="health-card">
        <div class="health-head">
          <span>{{ props.healthApp || 'gateway' }}</span>
          <span :class="['chip', props.healthOverall ? 'ok' : 'bad']">
            {{ props.healthOverall ? '健康' : '异常' }}
          </span>
        </div>
        <div class="dep-list">
          <div class="dep-row" v-for="dep in props.dependencies" :key="dep.name">
            <span>{{ dep.name }}</span>
            <span :class="['chip', dep.healthy && !dep.circuit_open ? 'ok' : 'bad']">
              {{ dep.healthy && !dep.circuit_open ? 'ok' : 'degraded' }}
            </span>
          </div>
        </div>
        <div class="ops">
          <button class="op-btn" :disabled="props.healthLoading" @click="emit('refreshHealth')">
            {{ props.healthLoading ? '刷新中...' : '刷新健康状态' }}
          </button>
          <button class="op-btn warn" :disabled="props.reindexLoading" @click="emit('reindex')">
            {{ props.reindexLoading ? '执行中...' : '重建索引' }}
          </button>
        </div>
        <div v-if="props.reindexInfo" class="reindex-info">{{ props.reindexInfo }}</div>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.right {
  padding: 14px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid var(--line-soft);
  box-shadow: var(--shadow-soft);
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-height: 0;
  overflow: hidden;
}

.right.closed {
  align-items: center;
  justify-content: center;
}

.toggle {
  align-self: flex-end;
  border: 1px solid rgba(27, 35, 32, 0.1);
  background: transparent;
  color: var(--ink-1);
  border-radius: 10px;
  padding: 6px 10px;
  cursor: pointer;
}

.panel-title {
  font-size: 12px;
  color: var(--ink-2);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.content {
  min-height: 0;
  overflow-y: auto;
  padding-right: 4px;
}

.control {
  display: grid;
  gap: 8px;
  margin-bottom: 10px;
  color: var(--ink-1);
}

.control input[type='range'] {
  width: 100%;
}

.control select {
  background: rgba(27, 35, 32, 0.04);
  border: 1px solid rgba(27, 35, 32, 0.1);
  color: var(--ink-0);
  padding: 8px 10px;
  border-radius: 10px;
}

.toggle-control {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.note {
  font-size: 12px;
  color: var(--ink-2);
  line-height: 1.6;
}

.health-card {
  border: 1px solid var(--line-soft);
  border-radius: 12px;
  padding: 10px;
  background: rgba(255, 255, 255, 0.92);
  display: grid;
  gap: 8px;
}

.health-head,
.dep-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}

.dep-list {
  display: grid;
  gap: 6px;
  max-height: 160px;
  overflow: auto;
}

.chip {
  font-size: 11px;
  border-radius: 999px;
  padding: 2px 8px;
}

.chip.ok {
  background: rgba(55, 155, 92, 0.15);
  color: #1e6d3f;
}

.chip.bad {
  background: rgba(190, 72, 72, 0.18);
  color: #8d1f1f;
}

.ops {
  display: grid;
  gap: 8px;
}

.op-btn {
  border: 1px solid var(--line-soft);
  background: #fff;
  border-radius: 10px;
  padding: 8px 10px;
  cursor: pointer;
}

.op-btn.warn {
  border-color: rgba(183, 139, 58, 0.45);
  background: rgba(183, 139, 58, 0.1);
}

.op-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.reindex-info {
  font-size: 12px;
  color: var(--ink-2);
}
</style>
