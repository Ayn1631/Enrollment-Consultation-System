<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import TopBarHeader from './components/TopBarHeader.vue'
import LeftSidebar from './components/LeftSidebar.vue'
import ChatMain from './components/ChatMain.vue'
import ActionBar from './components/ActionBar.vue'
import RightPanel from './components/RightPanel.vue'
import { getFeatures, getSavedSkills } from './services/api'
import { useStream } from './composables/useStream'
import { buildChatRequest, validateFeatureSelection } from './utils/requestBuilder'
import type { ChatMessage, ChatMode, ChatRequest, FeatureFlag, FeatureMeta, SavedSkill } from './types'

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
const selectedFeatures = ref<FeatureFlag[]>(['rag', 'citation_guard'])
const mode = ref<ChatMode>('chat')
const featureOptions = ref<FeatureMeta[]>([])
const savedSkills = ref<SavedSkill[]>([])
const savedSkillId = ref('')

const temperature = ref(0.6)
const topP = ref(0.9)
const model = ref('zyit-gpt')
const strictCitation = ref(true)
const rightOpen = ref(true)

const streamingText = ref('')
const streaming = ref(false)
const latestDegradedFeatures = ref<FeatureFlag[]>([])
let cancelStream: (() => void) | null = null

const modeLabel = computed(() => {
  if (mode.value === 'plan') return '规划执行'
  if (mode.value === 'guide') return '指引模式'
  return '对话模式'
})

const blockedReason = computed(() => {
  if (selectedFeatures.value.includes('use_saved_skill') && !savedSkillId.value) {
    return '已启用“使用以往技能”，请选择一个历史技能后再发送。'
  }
  return ''
})

const canSend = computed(() => !blockedReason.value)

const loadMeta = async () => {
  try {
    const [features, skills] = await Promise.all([getFeatures(), getSavedSkills()])
    featureOptions.value = features
    savedSkills.value = skills
    const defaultFeatures = features.filter((item) => item.default_enabled).map((item) => item.id)
    if (defaultFeatures.length) {
      selectedFeatures.value = defaultFeatures
    }
  } catch {
    featureOptions.value = [
      { id: 'rag', label: '本地RAG检索', default_enabled: true, dependencies: [] },
      { id: 'web_search', label: '联网搜索增强', default_enabled: false, dependencies: [] },
      { id: 'skill_exec', label: '通用技能执行', default_enabled: false, dependencies: [] },
      { id: 'use_saved_skill', label: '使用以往技能', default_enabled: false, dependencies: ['skill_exec'] },
      { id: 'citation_guard', label: '引用校验', default_enabled: true, dependencies: ['rag'] }
    ]
    savedSkills.value = [
      { id: 'admission_faq_v1', label: '招生FAQ助手', description: '聚焦招生政策与时间节点问答' }
    ]
  }
}

onMounted(() => {
  loadMeta()
})

const handleSend = async () => {
  const content = input.value.trim()
  if (!content || streaming.value) return
  const selectionError = validateFeatureSelection(selectedFeatures.value, savedSkillId.value)
  if (selectionError) {
    messages.value.push({
      id: newId(),
      role: 'system',
      content: selectionError,
      createdAt: new Date().toISOString()
    })
    return
  }

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

  const request: ChatRequest = buildChatRequest({
    sessionId: newId(),
    messages: messages.value.map((msg) => ({ role: msg.role, content: msg.content })),
    features: selectedFeatures.value,
    mode: mode.value,
    stream: true,
    savedSkillId: savedSkillId.value,
    strictCitation: strictCitation.value,
    temperature: temperature.value,
    topP: topP.value,
    model: model.value
  })

  const finalize = (done?: {
    status?: 'ok' | 'degraded' | 'failed'
    degraded_features?: FeatureFlag[]
    sources?: Array<{ title: string; url: string }>
    trace_id?: string
  }) => {
    if (streamingText.value.trim()) {
      latestDegradedFeatures.value = done?.degraded_features ?? []
      messages.value.push({
        id: newId(),
        role: 'assistant',
        content: streamingText.value.trim(),
        createdAt: new Date().toISOString(),
        status: done?.status ?? 'ok',
        degradedFeatures: done?.degraded_features ?? [],
        enabledFeatures: [...selectedFeatures.value],
        traceId: done?.trace_id,
        sources: done?.sources
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
      onDone: (done) => {
        finalize(done)
      },
      onError: () => {
        latestDegradedFeatures.value = []
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
    latestDegradedFeatures.value = []
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
      <LeftSidebar
        v-model:features="selectedFeatures"
        v-model:mode="mode"
        v-model:savedSkillId="savedSkillId"
        :feature-options="featureOptions"
        :saved-skills="savedSkills"
      />

      <main class="chat-area">
        <ChatMain
          :messages="messages"
          :streaming-text="streamingText"
          :is-streaming="streaming"
          :active-features="selectedFeatures"
          :degraded-features="latestDegradedFeatures"
        />
        <ActionBar
          v-model="input"
          :is-streaming="streaming"
          :can-send="canSend"
          :blocked-reason="blockedReason"
          @send="handleSend"
          @stop="handleStop"
        />
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
