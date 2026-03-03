<script setup lang="ts">
const props = defineProps<{ modelValue: string; isStreaming: boolean }>()
const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'send'): void
  (e: 'stop'): void
}>()

const updateValue = (value: string) => emit('update:modelValue', value)

const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    emit('send')
  }
}
</script>

<template>
  <div class="action-bar">
    <textarea
      class="input"
      :value="props.modelValue"
      placeholder="请输入你的招生咨询问题..."
      @input="updateValue(($event.target as HTMLTextAreaElement).value)"
      @keydown="handleKeydown"
      rows="2"
    ></textarea>
    <div class="actions">
      <button class="ghost">语音</button>
      <button class="ghost">附件</button>
      <button class="send" :disabled="props.isStreaming" @click="emit('send')">发送</button>
      <button v-if="props.isStreaming" class="stop" @click="emit('stop')">停止</button>
    </div>
  </div>
</template>

<style scoped>
.action-bar {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid var(--line-soft);
  border-radius: 18px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  box-shadow: var(--shadow-soft);
}

.input {
  width: 100%;
  border: none;
  resize: none;
  background: transparent;
  color: var(--ink-0);
  font-size: 14px;
  line-height: 1.6;
  outline: none;
}

.actions {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
  align-items: center;
}

.ghost {
  padding: 8px 12px;
  border-radius: 10px;
  background: rgba(27, 35, 32, 0.05);
  border: 1px solid rgba(27, 35, 32, 0.08);
  color: var(--ink-1);
  cursor: pointer;
}

.send {
  padding: 10px 18px;
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(214, 168, 91, 0.9), rgba(47, 143, 138, 0.75));
  border: none;
  color: #1a211f;
  font-weight: 600;
  cursor: pointer;
}

.stop {
  padding: 10px 14px;
  border-radius: 12px;
  border: 1px solid rgba(192, 75, 75, 0.4);
  background: rgba(192, 75, 75, 0.1);
  color: var(--ink-0);
  cursor: pointer;
}

.send:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
