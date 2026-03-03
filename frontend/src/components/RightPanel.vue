<script setup lang="ts">
const props = defineProps<{
  open: boolean
  temperature: number
  topP: number
  model: string
  strictCitation: boolean
}>()

const emit = defineEmits<{
  (e: 'toggle'): void
  (e: 'update:temperature', value: number): void
  (e: 'update:topP', value: number): void
  (e: 'update:model', value: string): void
  (e: 'update:strictCitation', value: boolean): void
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
    </div>
  </aside>
</template>

<style scoped>
.right {
  padding: 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid var(--line-soft);
  box-shadow: var(--shadow-soft);
  display: flex;
  flex-direction: column;
  gap: 16px;
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
  margin-bottom: 10px;
}

.control {
  display: grid;
  gap: 8px;
  margin-bottom: 12px;
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
</style>
