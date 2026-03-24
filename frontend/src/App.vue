<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import TopBarHeader from './components/TopBarHeader.vue'
import LeftSidebar from './components/LeftSidebar.vue'
import ChatMain from './components/ChatMain.vue'
import ActionBar from './components/ActionBar.vue'
import RightPanel from './components/RightPanel.vue'
import { compressMemory, getFeatures, getHealth, getSavedSkills, postReindex } from './services/api'
import { useStream } from './composables/useStream'
import { buildChatRequest, validateFeatureSelection } from './utils/requestBuilder'
import type {
  ChatMessage,
  ChatMode,
  ChatRequest,
  ChatSession,
  FeatureFlag,
  FeatureMeta,
  HealthDependency,
  SavedSkill
} from './types'

const { startStream } = useStream()

const newId = () => (crypto?.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`)
const DEFAULT_SESSION_TITLE = '新会话'

const createWelcomeMessage = (): ChatMessage => ({
  id: newId(),
  role: 'assistant',
  content: '欢迎来到中原工学院招生咨询系统。你可以直接提问，也可以开启工具模式获取更完整的答案。',
  createdAt: new Date().toISOString()
})

const createSession = (): ChatSession => {
  const now = new Date().toISOString()
  return {
    id: newId(),
    sessionId: newId(),
    title: DEFAULT_SESSION_TITLE,
    createdAt: now,
    updatedAt: now,
    messages: [createWelcomeMessage()],
    streamingText: '',
    isStreaming: false,
    latestDegradedFeatures: []
  }
}

const sessions = ref<ChatSession[]>([createSession()])
const activeSessionId = ref(sessions.value[0].id)

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
const healthLoading = ref(false)
const reindexLoading = ref(false)
const compressLoading = ref(false)
const healthApp = ref('')
const healthOverall = ref(true)
const healthDependencies = ref<HealthDependency[]>([])
const reindexInfo = ref('')
const compressInfo = ref('')
const pageNotice = ref<{ type: 'info' | 'warning' | 'error'; message: string } | null>(null)
let cancelStream: (() => void) | null = null
let pendingStreamText = ''
let streamFlushHandle: number | null = null
let currentStreamingSessionId: string | null = null

const activeSession = computed(() => {
  const matched = sessions.value.find((session) => session.id === activeSessionId.value)
  if (matched) {
    return matched
  }
  const fallback = sessions.value[0] ?? createSession()
  if (!sessions.value.length) {
    sessions.value = [fallback]
  }
  activeSessionId.value = fallback.id
  return fallback
})

const messages = computed(() => activeSession.value.messages)
const streamingText = computed(() => activeSession.value.streamingText)
const streaming = computed(() => activeSession.value.isStreaming)
const latestDegradedFeatures = computed(() => activeSession.value.latestDegradedFeatures)
const anyStreaming = computed(() => sessions.value.some((session) => session.isStreaming))

const modeLabel = computed(() => {
  if (mode.value === 'agent') return '智能体模式'
  if (mode.value === 'plan') return '规划执行'
  if (mode.value === 'guide') return '指引模式'
  return '对话模式'
})

const blockedReason = computed(() => {
  if (anyStreaming.value && !activeSession.value.isStreaming) {
    return '另一个会话正在生成回答，请等待这一轮结束后再继续发送。'
  }
  if (selectedFeatures.value.includes('use_saved_skill') && !savedSkillId.value) {
    return '已启用“使用以往技能”，请选择一个历史技能后再发送。'
  }
  return ''
})

const canSend = computed(() => !blockedReason.value)
const canCompressContext = computed(() => {
  const usefulMessages = activeSession.value.messages.filter((item) => item.role !== 'system' && item.content.trim())
  return usefulMessages.length >= 2 && !anyStreaming.value
})

const getErrorMessage = (error: unknown, fallback: string) => {
  if (error instanceof Error && error.message.trim()) {
    return error.message
  }
  return fallback
}

const showNotice = (message: string, type: 'info' | 'warning' | 'error' = 'error') => {
  pageNotice.value = { message, type }
}

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
    showNotice('获取后端功能配置失败，当前已切换为前端内置兜底配置。', 'warning')
  }
}

const refreshHealth = async () => {
  healthLoading.value = true
  try {
    const health = await getHealth()
    healthApp.value = health.app
    healthOverall.value = health.healthy
    healthDependencies.value = health.dependencies
  } catch {
    healthApp.value = 'gateway'
    healthOverall.value = false
    healthDependencies.value = [
      { name: 'api-gateway', healthy: false, circuit_open: false, last_error: '无法连接后端' }
    ]
    showNotice('后端健康检查失败，请确认后端服务已经启动并且接口地址配置正确。', 'error')
  } finally {
    healthLoading.value = false
  }
}

const triggerReindex = async () => {
  reindexLoading.value = true
  try {
    const result = await postReindex()
    reindexInfo.value = `重建完成，当前索引块数：${result.result.chunks}`
    await refreshHealth()
  } catch (error) {
    reindexInfo.value = '重建索引失败，请检查后端服务状态。'
    showNotice(getErrorMessage(error, '重建索引失败，请检查后端服务状态。'), 'error')
  } finally {
    reindexLoading.value = false
  }
}

onMounted(() => {
  void Promise.all([loadMeta(), refreshHealth()])
})

const getSessionById = (sessionId: string) => sessions.value.find((session) => session.id === sessionId) ?? null

const touchSession = (sessionId: string) => {
  const index = sessions.value.findIndex((session) => session.id === sessionId)
  if (index === -1) return
  const [session] = sessions.value.splice(index, 1)
  if (!session) return
  session.updatedAt = new Date().toISOString()
  sessions.value.unshift(session)
}

const buildSessionTitle = (content: string) => {
  const compact = content.trim().replace(/\s+/g, ' ')
  if (!compact) return DEFAULT_SESSION_TITLE
  return compact.length > 16 ? `${compact.slice(0, 16)}...` : compact
}

const flushStreamingText = () => {
  streamFlushHandle = null
  if (!pendingStreamText) return
  if (!currentStreamingSessionId) {
    pendingStreamText = ''
    return
  }
  const session = getSessionById(currentStreamingSessionId)
  if (!session) {
    pendingStreamText = ''
    return
  }
  session.streamingText += pendingStreamText
  touchSession(session.id)
  pendingStreamText = ''
}

const queueStreamingDelta = (delta: string) => {
  pendingStreamText += delta
  if (streamFlushHandle !== null) return
  streamFlushHandle = requestAnimationFrame(flushStreamingText)
}

const drainStreamingBuffer = () => {
  if (streamFlushHandle !== null) {
    cancelAnimationFrame(streamFlushHandle)
    streamFlushHandle = null
  }
  if (!pendingStreamText) return
  if (currentStreamingSessionId) {
    const session = getSessionById(currentStreamingSessionId)
    if (session) {
      session.streamingText += pendingStreamText
      touchSession(session.id)
    }
  }
  pendingStreamText = ''
}

const handleSend = async () => {
  const content = input.value.trim()
  const session = activeSession.value
  if (!content || anyStreaming.value) return
  const selectionError = validateFeatureSelection(selectedFeatures.value, savedSkillId.value)
  if (selectionError) {
    session.messages.push({
      id: newId(),
      role: 'system',
      content: selectionError,
      createdAt: new Date().toISOString()
    })
    touchSession(session.id)
    return
  }

  const userMessage: ChatMessage = {
    id: newId(),
    role: 'user',
    content,
    createdAt: new Date().toISOString()
  }

  session.messages.push(userMessage)
  if (session.title === DEFAULT_SESSION_TITLE) {
    session.title = buildSessionTitle(content)
  }
  touchSession(session.id)
  input.value = ''
  session.isStreaming = true
  session.streamingText = ''
  session.latestDegradedFeatures = []
  currentStreamingSessionId = session.id

  const request: ChatRequest = buildChatRequest({
    sessionId: session.sessionId,
    messages: session.messages.map((msg) => ({ role: msg.role, content: msg.content })),
    features: selectedFeatures.value,
    mode: mode.value,
    stream: true,
    savedSkillId: savedSkillId.value,
    strictCitation: strictCitation.value,
    temperature: temperature.value,
    topP: topP.value,
    model: model.value
  })
  console.log('[App.handleSend] request built', {
    session_id: request.session_id,
    features: request.features,
    mode: request.mode,
    strict_citation: request.strict_citation,
    model: request.model,
    selectedFeatures: selectedFeatures.value
  })

  const finalize = (done?: {
    status?: 'ok' | 'degraded' | 'failed'
    degraded_features?: FeatureFlag[]
    sources?: Array<{ title: string; url: string }>
    trace_id?: string
  }) => {
    drainStreamingBuffer()
    const targetSession = getSessionById(session.id)
    if (!targetSession) return
    console.log('[App.handleSend] finalize', {
      done,
      buffered_length: targetSession.streamingText.length,
      degraded_features: done?.degraded_features ?? []
    })
    if (targetSession.streamingText.trim()) {
      targetSession.latestDegradedFeatures = done?.degraded_features ?? []
      targetSession.messages.push({
        id: newId(),
        role: 'assistant',
        content: targetSession.streamingText.trim(),
        createdAt: new Date().toISOString(),
        status: done?.status ?? 'ok',
        degradedFeatures: done?.degraded_features ?? [],
        enabledFeatures: [...selectedFeatures.value],
        traceId: done?.trace_id,
        sources: done?.sources
      })
    }
    targetSession.streamingText = ''
    targetSession.isStreaming = false
    touchSession(targetSession.id)
    cancelStream = null
    if (currentStreamingSessionId === targetSession.id) {
      currentStreamingSessionId = null
    }
  }

  try {
    pageNotice.value = null
    cancelStream = await startStream(request, {
      onDelta: (delta) => {
        console.log('[App.handleSend] delta', delta)
        queueStreamingDelta(delta)
      },
      onDone: (done) => {
        console.log('[App.handleSend] stream done', done)
        finalize(done)
      },
      onError: (error) => {
        console.error('[App.handleSend] stream error', error)
        const message = getErrorMessage(error, '系统连接异常，请稍后重试。')
        const targetSession = getSessionById(session.id)
        targetSession?.messages.push({
          id: newId(),
          role: 'system',
          content: message,
          createdAt: new Date().toISOString()
        })
        if (targetSession) {
          targetSession.latestDegradedFeatures = []
          touchSession(targetSession.id)
        }
        showNotice(message, 'error')
        finalize()
      }
    })
  } catch (error) {
    console.error('[App.handleSend] request error', error)
    const message = getErrorMessage(error, '无法连接后端服务，请检查接口配置。')
    session.messages.push({
      id: newId(),
      role: 'system',
      content: message,
      createdAt: new Date().toISOString()
    })
    session.latestDegradedFeatures = []
    touchSession(session.id)
    showNotice(message, 'error')
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

const handleCreateSession = () => {
  const session = createSession()
  sessions.value.unshift(session)
  activeSessionId.value = session.id
  input.value = ''
}

const handleSwitchSession = (sessionId: string) => {
  if (activeSessionId.value === sessionId) return
  activeSessionId.value = sessionId
  pageNotice.value = null
}

const handleClearSession = () => {
  const session = activeSession.value
  if (session.id === currentStreamingSessionId && cancelStream) {
    cancelStream()
    cancelStream = null
    currentStreamingSessionId = null
    pendingStreamText = ''
    if (streamFlushHandle !== null) {
      cancelAnimationFrame(streamFlushHandle)
      streamFlushHandle = null
    }
  }
  const now = new Date().toISOString()
  session.sessionId = newId()
  session.title = DEFAULT_SESSION_TITLE
  session.createdAt = now
  session.updatedAt = now
  session.messages = [createWelcomeMessage()]
  session.streamingText = ''
  session.isStreaming = false
  session.latestDegradedFeatures = []
  input.value = ''
  compressInfo.value = ''
  showNotice('当前会话已清空，并已重置为新的空白对话。', 'info')
}

const handleCompressContext = async () => {
  if (!canCompressContext.value) {
    showNotice('当前会话内容太少，或仍在生成中，暂时不能压缩上下文。', 'warning')
    return
  }
  const session = activeSession.value
  compressLoading.value = true
  compressInfo.value = ''
  try {
    const response = await compressMemory({
      session_id: session.sessionId,
      session_title: session.title,
      messages: session.messages.map((item) => ({
        role: item.role,
        content: item.content
      }))
    })
    compressInfo.value =
      `已写入长期记忆 ${response.long_memory_count} 条，特殊记忆 ${response.special_memory_count} 条。`
    const noteText = response.notes.length ? `备注：${response.notes.join('；')}` : ''
    const summaryText = response.long_summary ? `摘要：${response.long_summary}` : '本次未生成长期摘要。'
    showNotice(
      `上下文压缩完成，模型 ${response.model || 'unknown'}。${summaryText}${noteText ? ` ${noteText}` : ''}`,
      'info'
    )
  } catch (error) {
    const message = getErrorMessage(error, '压缩上下文失败，请检查后端服务。')
    compressInfo.value = '压缩失败，请查看页面提示或后端日志。'
    showNotice(message, 'error')
  } finally {
    compressLoading.value = false
  }
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
        :sessions="sessions"
        :active-session-id="activeSessionId"
        @switch-session="handleSwitchSession"
        @create-session="handleCreateSession"
        @clear-session="handleClearSession"
      />

      <main class="chat-area">
        <div v-if="pageNotice" class="page-notice" :class="pageNotice.type">
          <span>{{ pageNotice.message }}</span>
          <button type="button" class="notice-close" @click="pageNotice = null">知道了</button>
        </div>
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
        :health-loading="healthLoading"
        :reindex-loading="reindexLoading"
        :compress-loading="compressLoading"
        :health-app="healthApp"
        :health-overall="healthOverall"
        :dependencies="healthDependencies"
        :reindex-info="reindexInfo"
        :compress-info="compressInfo"
        :can-compress-context="canCompressContext"
        @refresh-health="refreshHealth"
        @reindex="triggerReindex"
        @compress-context="handleCompressContext"
        @toggle="toggleRightPanel"
      />
    </div>
  </div>
</template>

<style scoped>
.app-shell {
  height: 100vh;
  padding: 14px 16px 18px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  animation: rise-in 0.8s ease both;
  overflow: hidden;
}

.layout {
  flex: 1;
  display: grid;
  grid-template-columns: 248px minmax(0, 1fr) 296px;
  gap: 14px;
  align-items: stretch;
  min-height: 0;
  overflow: hidden;
}

.chat-area {
  background: var(--panel);
  border: 1px solid var(--line-soft);
  border-radius: 22px;
  padding: 16px 16px 14px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  box-shadow: var(--shadow-soft);
  backdrop-filter: blur(14px);
  min-height: 0;
  overflow: hidden;
}

.page-notice {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border-radius: 14px;
  font-size: 13px;
  line-height: 1.5;
}

.page-notice.info {
  background: rgba(31, 86, 166, 0.1);
  border: 1px solid rgba(31, 86, 166, 0.16);
  color: #17427d;
}

.page-notice.warning {
  background: rgba(166, 30, 36, 0.08);
  border: 1px solid rgba(166, 30, 36, 0.16);
  color: var(--accent-cool);
}

.page-notice.error {
  background: rgba(183, 28, 28, 0.1);
  border: 1px solid rgba(183, 28, 28, 0.18);
  color: #8b1d1d;
}

.notice-close {
  border: none;
  background: transparent;
  color: inherit;
  font: inherit;
  cursor: pointer;
  white-space: nowrap;
}

.layout.compact {
  grid-template-columns: 248px minmax(0, 1fr) 76px;
}

@media (max-width: 1280px) {
  .layout {
    grid-template-columns: 216px minmax(0, 1fr) 264px;
  }
}

@media (max-width: 1080px) {
  .app-shell {
    padding: 12px;
    gap: 12px;
  }

  .layout {
    grid-template-columns: 1fr;
  }

  .chat-area {
    min-height: 0;
  }
}
</style>
