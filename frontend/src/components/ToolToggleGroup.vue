<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import type { FeatureFlag, FeatureMeta } from '../types'

const props = defineProps<{ modelValue: FeatureFlag[]; options: FeatureMeta[] }>()
const emit = defineEmits<{ (e: 'update:modelValue', value: FeatureFlag[]): void }>()

const open = ref(false)
const dropdownRef = ref<HTMLElement | null>(null)

const currentLabel = computed(() => {
  if (!props.modelValue.length) return '请选择功能'
  const labels = props.options
    .filter((item) => props.modelValue.includes(item.id))
    .map((item) => item.label || item.id)
  return labels.join(' / ')
})

const toggleMenu = () => {
  open.value = !open.value
}

const closeMenu = () => {
  open.value = false
}

const toggle = (value: FeatureFlag) => {
  const next = props.modelValue.includes(value)
    ? props.modelValue.filter((item) => item !== value)
    : [...props.modelValue, value]
  emit('update:modelValue', next)
}

const handleClickOutside = (event: MouseEvent) => {
  if (!dropdownRef.value) return
  if (!dropdownRef.value.contains(event.target as Node)) {
    closeMenu()
  }
}

onMounted(() => document.addEventListener('click', handleClickOutside))
onBeforeUnmount(() => document.removeEventListener('click', handleClickOutside))
</script>

<template>
  <div class="dropdown" ref="dropdownRef">
    <button class="dropdown-trigger" type="button" @click.stop="toggleMenu">
      <div>
        <div class="trigger-title">功能权限</div>
        <div class="trigger-value">{{ currentLabel }}</div>
      </div>
      <span class="chevron" :class="{ open }">▾</span>
    </button>

    <div v-if="open" class="dropdown-menu">
      <button
        v-for="option in props.options"
        :key="option.id"
        type="button"
        class="dropdown-item"
        :class="{ active: props.modelValue.includes(option.id) }"
        @click="toggle(option.id)"
      >
        <span class="check">{{ props.modelValue.includes(option.id) ? '✓' : '' }}</span>
        <span class="text">
          <span class="label">{{ option.label || option.id }}</span>
          <span class="hint">{{
            option.dependencies?.length ? `依赖: ${option.dependencies.join(', ')}` : '可独立启用'
          }}</span>
        </span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.dropdown {
  position: relative;
}

.dropdown-trigger {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px;
  border-radius: 14px;
  border: 1px solid var(--line-soft);
  background: rgba(255, 255, 255, 0.9);
  color: var(--ink-0);
  cursor: pointer;
  box-shadow: var(--shadow-soft);
  transition: 0.2s ease;
  text-align: left;
}

.dropdown-trigger:hover {
  transform: translateY(-1px);
}

.trigger-title {
  font-size: 12px;
  color: var(--ink-2);
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.trigger-value {
  font-size: 14px;
  font-weight: 600;
  margin-top: 4px;
}

.chevron {
  font-size: 16px;
  color: var(--ink-2);
  transition: transform 0.2s ease;
}

.chevron.open {
  transform: rotate(180deg);
}

.dropdown-menu {
  position: absolute;
  top: calc(100% + 8px);
  left: 0;
  right: 0;
  border-radius: 14px;
  border: 1px solid var(--line-soft);
  background: #ffffff;
  box-shadow: var(--shadow-strong);
  padding: 6px;
  z-index: 10;
}

.dropdown-item {
  width: 100%;
  display: grid;
  grid-template-columns: 20px 1fr;
  gap: 10px;
  align-items: center;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid transparent;
  background: transparent;
  text-align: left;
  cursor: pointer;
  transition: 0.18s ease;
  color: var(--ink-0);
}

.dropdown-item:hover {
  background: rgba(183, 139, 58, 0.1);
  border-color: rgba(183, 139, 58, 0.2);
}

.dropdown-item.active {
  background: rgba(47, 143, 138, 0.12);
  border-color: rgba(47, 143, 138, 0.3);
}

.check {
  font-size: 14px;
  color: var(--accent);
}

.label {
  display: block;
  font-size: 14px;
  font-weight: 600;
}

.hint {
  display: block;
  font-size: 12px;
  color: var(--ink-2);
  margin-top: 2px;
}
</style>
