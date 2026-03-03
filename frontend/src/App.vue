<script setup lang="ts">
import { computed, ref } from 'vue'
import TopBarHeader from './components/TopBarHeader.vue'
import LeftSidebar from './components/LeftSidebar.vue'
import ChatMain from './components/ChatMain.vue'
import ActionBar from './components/ActionBar.vue'
import RightPanel from './components/RightPanel.vue'
import { useStream } from './composables/useStream'
import type { ChatMessage, ChatMode, ChatRequest, ToolMode } from './types'

const { startStream } = useStream()

const newId = () => (crypto?.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`)

const messages = ref<ChatMessage[]>([
  {
    id: newId(),
    role: 'assistant',
    content:
      '欢迎来到中原工学院招生咨询系统。你可以直接提问，也可以开启工具模式获取更完整的答案。',
    createdAt: new Date().toISOString()
  }
])

const input = ref('')
const selectedTools = ref<ToolMode[]>(['search'])
const mode = ref<ChatMode>('chat')

const temperature = ref(0.6)
const topP = ref(0.9)
const model = ref('zyit-gpt')
const strictCitation = ref(false)
const rightOpen = ref(true)

const streamingText = ref('')
const streaming = ref(false)
let cancelStream: (() => void) | null = null

const modeLabel = computed(() => {
  if (mode.value === 'plan') return '规划执行'
  if (mode.value === 'guide') return '指引模式'
  return '对话模式'
})

const handleSend = async () => {
  const content = input.value.trim()
  if (!content || streaming.value) return

  const userMessage: ChatMessage = {
    id: newId(),
    role: 'user',
    content,
    createdAt: new Date().toISOString()
  }

  messages.value.push(userMessage)
  input.value = ''
  streaming.value = true
  streamingText.value = ''

  const request: ChatRequest = {
    session_id: newId(),
    messages: messages.value.map((msg) => ({ role: msg.role, content: msg.content })),
    tools: selectedTools.value,
    mode: mode.value,
    stream: true
  }

  const finalize = () => {
    if (streamingText.value.trim()) {
      messages.value.push({
        id: newId(),
        role: 'assistant',
        content: streamingText.value.trim(),
        createdAt: new Date().toISOString()
      })
    }
    streamingText.value = ''
    streaming.value = false
    cancelStream = null
  }

  try {
    cancelStream = await startStream(request, {
      onDelta: (delta) => {
        streamingText.value += delta
      },
      onDone: () => {
        finalize()
      },
      onError: () => {
        messages.value.push({
          id: newId(),
          role: 'system',
          content: '系统连接异常，请稍后重试。',
          createdAt: new Date().toISOString()
        })
        finalize()
      }
    })
  } catch (error) {
    messages.value.push({
      id: newId(),
      role: 'system',
      content: '无法连接后端服务，请检查接口配置。',
      createdAt: new Date().toISOString()
    })
    finalize()
  }
}

const handleStop = () => {
  if (cancelStream) {
    cancelStream()
  }
}

const toggleRightPanel = () => {
  rightOpen.value = !rightOpen.value
}
</script>

<template>
  <div class="app-shell">
    <TopBarHeader :mode="modeLabel" />

    <div class="layout" :class="{ compact: !rightOpen }">
      <LeftSidebar v-model:tools="selectedTools" v-model:mode="mode" />

      <main class="chat-area">
        <ChatMain :messages="messages" :streaming-text="streamingText" :is-streaming="streaming" />
        <ActionBar v-model="input" :is-streaming="streaming" @send="handleSend" @stop="handleStop" />
      </main>

      <RightPanel
        :open="rightOpen"
        v-model:temperature="temperature"
        v-model:topP="topP"
        v-model:model="model"
        v-model:strictCitation="strictCitation"
        @toggle="toggleRightPanel"
      />
    </div>
  </div>
</template>

<style scoped>
.app-shell {
  min-height: 100vh;
  padding: 18px 20px 26px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  animation: rise-in 0.8s ease both;
}

.layout {
  flex: 1;
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr) 320px;
  gap: 16px;
  align-items: stretch;
}

.chat-area {
  background: var(--panel);
  border: 1px solid var(--line-soft);
  border-radius: 22px;
  padding: 18px 18px 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  box-shadow: var(--shadow-soft);
  backdrop-filter: blur(14px);
}


.layout.compact {
  grid-template-columns: 260px minmax(0, 1fr) 80px;
}

@media (max-width: 1280px) {
  .layout {
    grid-template-columns: 220px minmax(0, 1fr) 280px;
  }
}

@media (max-width: 1080px) {
  .layout {
    grid-template-columns: 1fr;
  }
  .chat-area {
    min-height: 70vh;
  }
}
</style>
