<script setup lang="ts">
const props = defineProps<{ modelValue: string; isStreaming: boolean; canSend?: boolean; blockedReason?: string }>()
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
      <button class="send" :disabled="props.isStreaming || props.canSend === false" @click="emit('send')">发送</button>
      <button v-if="props.isStreaming" class="stop" @click="emit('stop')">停止</button>
    </div>
    <div v-if="props.canSend === false && props.blockedReason" class="blocked-tip">{{ props.blockedReason }}</div>
  </div>
</template>

<style scoped>
.action-bar {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid var(--line-soft);
  border-radius: 18px;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  box-shadow: var(--shadow-soft);
  flex: 0 0 auto;
}

.input {
  width: 100%;
  border: none;
  resize: none;
  background: transparent;
  color: var(--ink-0);
  font-size: 14px;
  line-height: 1.55;
  outline: none;
  min-height: 52px;
  max-height: 120px;
}

.actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  align-items: center;
  flex-wrap: wrap;
}

.ghost {
  padding: 7px 11px;
  border-radius: 10px;
  background: rgba(27, 35, 32, 0.05);
  border: 1px solid rgba(27, 35, 32, 0.08);
  color: var(--ink-1);
  cursor: pointer;
}

.send {
  padding: 9px 16px;
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(198, 40, 50, 0.96), rgba(127, 21, 27, 0.9));
  border: none;
  color: #fff7f7;
  font-weight: 600;
  cursor: pointer;
}

.stop {
  padding: 9px 13px;
  border-radius: 12px;
  border: 1px solid rgba(166, 30, 36, 0.4);
  background: rgba(166, 30, 36, 0.12);
  color: var(--ink-0);
  cursor: pointer;
}

.send:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.blocked-tip {
  font-size: 12px;
  color: var(--accent-cool);
  padding-top: 2px;
}
</style>
