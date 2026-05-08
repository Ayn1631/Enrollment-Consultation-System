<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { editorialBriefings, modeOptions, promptSeeds, type SupportedMode } from './consoleMeta'
import MarkdownContent from './components/MarkdownContent.vue'
import { compressMemory, getHealth, postReindex, startChatStream } from './services/api'
import type {
  AgentStepEvent,
  AgentStrategy,
  ChatMessage,
  ChatRequest,
  ChatSession,
  ChatSource,
  FeatureFlag,
  HealthDependency
} from './types'

type NoticeTone = 'info' | 'warning' | 'error'

const modelOptions = ['gpt-5.4', 'gpt-5.5']
const DEFAULT_SESSION_TITLE = '新会话'

const newId = () => (crypto?.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`)

const createWelcomeMessage = (): ChatMessage => ({
  id: newId(),
  role: 'assistant',
  content:
    '欢迎使用中原工学院招生咨询系统。你可以直接提问，也可以切换到 RAG 问答或专家模式，让系统在资料和步骤链路上给出更稳的回答。',
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
    latestDegradedFeatures: [],
    currentAgentTrace: []
  }
}

const sessions = ref<ChatSession[]>([createSession()])
const activeSessionId = ref(sessions.value[0].id)
const input = ref('')
const mode = ref<SupportedMode>('chat')
const temperature = ref(0.45)
const topP = ref(0.85)
const model = ref('gpt-5.4')
const agentStrategy = ref<AgentStrategy>('quality')
const rightOpen = ref(true)
const healthLoading = ref(false)
const reindexLoading = ref(false)
const compressLoading = ref(false)
const healthApp = ref('')
const healthOverall = ref(true)
const healthDependencies = ref<HealthDependency[]>([])
const reindexInfo = ref('')
const compressInfo = ref('')
const pageNotice = ref<{ type: NoticeTone; message: string } | null>(null)
const messagesRef = ref<HTMLElement | null>(null)
const textareaRef = ref<HTMLTextAreaElement | null>(null)

let cancelStream: (() => void) | null = null
let pendingStreamText = ''
let streamFlushHandle: number | null = null
let currentStreamingSessionId: string | null = null

const activeSession = computed(() => {
  const matched = sessions.value.find((session) => session.id === activeSessionId.value)
  if (matched) return matched
  const fallback = sessions.value[0] ?? createSession()
  if (!sessions.value.length) {
    sessions.value = [fallback]
  }
  activeSessionId.value = fallback.id
  return fallback
})

const modeMeta = computed(() => modeOptions.find((item) => item.value === mode.value) ?? modeOptions[0])
const featuredPrompts = computed(() => promptSeeds[mode.value])
const messages = computed(() => activeSession.value.messages)
const streamingText = computed(() => activeSession.value.streamingText)
const streaming = computed(() => activeSession.value.isStreaming)
const latestDegradedFeatures = computed(() => activeSession.value.latestDegradedFeatures)
const anyStreaming = computed(() => sessions.value.some((session) => session.isStreaming))
const canCompressContext = computed(() => {
  const usefulMessages = activeSession.value.messages.filter((item) => item.role !== 'system' && item.content.trim())
  return usefulMessages.length >= 2 && !anyStreaming.value
})
const blockedReason = computed(() => {
  if (anyStreaming.value && !activeSession.value.isStreaming) {
    return '另一个会话正在生成回答，请等这一轮结束。'
  }
  return ''
})
const canSend = computed(() => !blockedReason.value)
const healthStateLabel = computed(() => (healthOverall.value ? '系统在线' : '服务波动'))
const todayLabel = computed(() =>
  new Intl.DateTimeFormat('zh-CN', { month: 'long', day: 'numeric', weekday: 'long' }).format(new Date())
)
const sourceCount = computed(() =>
  messages.value.reduce((total, message) => total + (message.sources?.length ?? 0), 0)
)
const briefingItems = computed(() => editorialBriefings[mode.value])
const latestAgentTraceMessage = computed(
  () =>
    [...messages.value]
      .reverse()
      .find((message) => message.role === 'assistant' && Boolean(message.agentTrace?.length)) ?? null
)
const visibleAgentTrace = computed(() =>
  activeSession.value.currentAgentTrace.length
    ? activeSession.value.currentAgentTrace
    : (latestAgentTraceMessage.value?.agentTrace ?? [])
)
const traceBandTitle = computed(() => (activeSession.value.currentAgentTrace.length ? '当前执行轨迹' : '最近一轮专家轨迹'))
const traceBandNote = computed(() =>
  activeSession.value.currentAgentTrace.length ? '实时展示当前步骤推进情况。' : '保留最近一次专家模式执行链路，方便回看。'
)
const hasOnlyWelcomeMessage = computed(
  () =>
    messages.value.length === 1 &&
    messages.value[0]?.role === 'assistant' &&
    !messages.value[0]?.sources?.length &&
    !streaming.value
)

const getErrorMessage = (error: unknown, fallback: string) => {
  if (error instanceof Error && error.message.trim()) {
    return error.message
  }
  return fallback
}

const resolveModeFeatures = (targetMode: SupportedMode): FeatureFlag[] => {
  if (targetMode === 'agent') return ['rag', 'web_search', 'skill_exec', 'citation_guard']
  if (targetMode === 'rag') return ['rag', 'citation_guard']
  return []
}

const buildChatRequest = (session: ChatSession): ChatRequest => ({
  session_id: session.sessionId,
  messages: session.messages.map((item) => ({ role: item.role, content: item.content })),
  features: resolveModeFeatures(mode.value),
  mode: mode.value,
  stream: true,
  strict_citation: mode.value === 'rag' || mode.value === 'agent',
  temperature: temperature.value,
  top_p: topP.value,
  model: model.value,
  agent_strategy: agentStrategy.value
})

const showNotice = (message: string, type: NoticeTone = 'error') => {
  pageNotice.value = { message, type }
}

const formatTime = (value: string) =>
  new Date(value).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit'
  })

const getSessionSummary = (session: ChatSession) => {
  const lastMessage =
    [...session.messages].reverse().find((item) => item.role !== 'system') ??
    session.messages[session.messages.length - 1]
  if (!lastMessage) return '暂无消息'
  return lastMessage.content.trim().replace(/\s+/g, ' ').slice(0, 36) || '暂无消息'
}

const roleLabel = (role: ChatMessage['role']) => {
  if (role === 'user') return '提问方'
  if (role === 'assistant') return '系统回答'
  return '系统提示'
}

const isLocalKnowledgeSource = (source: ChatSource) => {
  const url = source.url.trim()
  if (!url) return true
  if (/^https?:\/\//i.test(url)) return false
  if (/^[a-z]:\\/i.test(url)) return true
  if (url.startsWith('docs/') || url.startsWith('docs\\')) return true
  if (url.startsWith('./') || url.startsWith('../')) return true
  return !/^[a-z][a-z0-9+.-]*:/i.test(url)
}

const touchSession = (sessionId: string) => {
  const index = sessions.value.findIndex((session) => session.id === sessionId)
  if (index === -1) return
  const [session] = sessions.value.splice(index, 1)
  if (!session) return
  session.updatedAt = new Date().toISOString()
  sessions.value.unshift(session)
}

const getSessionById = (sessionId: string) => sessions.value.find((session) => session.id === sessionId) ?? null

const buildSessionTitle = (content: string) => {
  const compact = content.trim().replace(/\s+/g, ' ')
  if (!compact) return DEFAULT_SESSION_TITLE
  return compact.length > 18 ? `${compact.slice(0, 18)}...` : compact
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
    showNotice('后端健康检查失败，请确认服务已经启动且接口地址配置正确。', 'error')
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

const scrollToBottom = async () => {
  await nextTick()
  const container = messagesRef.value
  if (!container) return
  container.scrollTop = container.scrollHeight
}

const handleSend = async () => {
  const content = input.value.trim()
  const session = activeSession.value
  if (!content || anyStreaming.value) return

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
  session.currentAgentTrace = []
  currentStreamingSessionId = session.id

  const request = buildChatRequest(session)
  const requestFeatures = [...request.features]
  const hasToolFeatures = requestFeatures.length > 0

  const finalize = (done?: {
    status?: 'ok' | 'degraded' | 'failed'
    degraded_features?: FeatureFlag[]
    sources?: Array<{ title: string; url: string }>
    trace_id?: string
    tool_audit?: string[]
    error_message?: string
  }) => {
    drainStreamingBuffer()
    const targetSession = getSessionById(session.id)
    if (!targetSession) return

    const failureText =
      done?.status === 'failed'
        ? [done.error_message ? `失败原因：${done.error_message}` : '', done.trace_id ? `trace_id：${done.trace_id}` : '']
            .filter(Boolean)
            .join('\n')
        : ''

    if (targetSession.streamingText.trim()) {
      targetSession.latestDegradedFeatures = hasToolFeatures ? (done?.degraded_features ?? []) : []
      targetSession.messages.push({
        id: newId(),
        role: 'assistant',
        content: targetSession.streamingText.trim(),
        createdAt: new Date().toISOString(),
        status: done?.status ?? 'ok',
        degradedFeatures: hasToolFeatures ? (done?.degraded_features ?? []) : [],
        enabledFeatures: requestFeatures,
        traceId: hasToolFeatures ? done?.trace_id : undefined,
        sources: hasToolFeatures ? done?.sources : undefined,
        errorMessage: done?.error_message,
        toolAudit: hasToolFeatures ? (done?.tool_audit ?? []) : [],
        agentTrace: hasToolFeatures ? [...targetSession.currentAgentTrace] : []
      })
    } else if (failureText) {
      targetSession.messages.push({
        id: newId(),
        role: 'assistant',
        content: failureText,
        createdAt: new Date().toISOString(),
        status: 'failed',
        degradedFeatures: hasToolFeatures ? (done?.degraded_features ?? []) : [],
        enabledFeatures: requestFeatures,
        traceId: hasToolFeatures ? done?.trace_id : undefined,
        sources: hasToolFeatures ? done?.sources : undefined,
        errorMessage: done?.error_message,
        toolAudit: hasToolFeatures ? (done?.tool_audit ?? []) : [],
        agentTrace: hasToolFeatures ? [...targetSession.currentAgentTrace] : []
      })
    }

    targetSession.streamingText = ''
    targetSession.isStreaming = false
    targetSession.currentAgentTrace = []
    touchSession(targetSession.id)

    if (done?.status === 'failed') {
      const detail = done.error_message ? `失败原因：${done.error_message}` : '本轮执行失败。'
      const traceText = done.trace_id ? ` trace_id：${done.trace_id}` : ''
      showNotice(`${detail}${traceText}`, 'error')
    }

    cancelStream = null
    if (currentStreamingSessionId === targetSession.id) {
      currentStreamingSessionId = null
    }
  }

  try {
    pageNotice.value = null
    cancelStream = await startChatStream(request, {
      onDelta: (delta) => {
        queueStreamingDelta(delta)
      },
      onStep: (event: AgentStepEvent) => {
        const targetSession = getSessionById(session.id)
        if (!targetSession) return
        targetSession.currentAgentTrace = [...targetSession.currentAgentTrace, event]
        touchSession(targetSession.id)
      },
      onDone: (done) => {
        finalize(done)
      },
      onError: (error) => {
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
  if (!cancelStream || !currentStreamingSessionId) return
  cancelStream()
  drainStreamingBuffer()
  const session = getSessionById(currentStreamingSessionId)
  if (!session) {
    cancelStream = null
    currentStreamingSessionId = null
    return
  }
  if (session.streamingText.trim()) {
    session.messages.push({
      id: newId(),
      role: 'assistant',
      content: session.streamingText.trim(),
      createdAt: new Date().toISOString(),
      status: 'ok',
      degradedFeatures: [],
      enabledFeatures: resolveModeFeatures(mode.value),
      agentTrace: [...session.currentAgentTrace]
    })
  }
  session.streamingText = ''
  session.isStreaming = false
  session.currentAgentTrace = []
  session.latestDegradedFeatures = []
  touchSession(session.id)
  cancelStream = null
  currentStreamingSessionId = null
  showNotice('已停止本轮回答。', 'info')
}

const handleCreateSession = () => {
  const session = createSession()
  sessions.value.unshift(session)
  activeSessionId.value = session.id
  input.value = ''
  pageNotice.value = null
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
  session.currentAgentTrace = []
  input.value = ''
  compressInfo.value = ''
  showNotice('当前会话已清空，新的空白对话已经就位。', 'info')
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
    compressInfo.value = `已写入长期记忆 ${response.long_memory_count} 条，特殊记忆 ${response.special_memory_count} 条。`
    const noteText = response.notes.length ? `备注：${response.notes.join('；')}` : ''
    const summaryText = response.long_summary ? `摘要：${response.long_summary}` : '本次未生成长期摘要。'
    showNotice(`上下文压缩完成。${summaryText}${noteText ? ` ${noteText}` : ''}`, 'info')
  } catch (error) {
    const message = getErrorMessage(error, '压缩上下文失败，请检查后端服务。')
    compressInfo.value = '压缩失败，请查看页面提示或后端日志。'
    showNotice(message, 'error')
  } finally {
    compressLoading.value = false
  }
}

const fillPrompt = async (prompt: string) => {
  input.value = prompt
  await nextTick()
  textareaRef.value?.focus()
}

const handleComposerKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    void handleSend()
  }
}

watch(
  () => [activeSessionId.value, messages.value.length, streamingText.value, streaming.value],
  () => {
    void scrollToBottom()
  },
  { flush: 'post' }
)

onMounted(() => {
  void refreshHealth()
})
</script>

<template>
  <div class="shell">
    <div class="shell__wash" aria-hidden="true"></div>
    <div class="shell__grain" aria-hidden="true"></div>

    <header class="masthead">
      <div class="masthead__copy">
        <p class="masthead__eyebrow">Admissions Intelligence Console</p>
        <h1>中原工学院招生咨询系统</h1>
        <p class="masthead__deck">{{ modeMeta.description }}</p>
      </div>

      <div class="masthead__aside">
        <div class="masthead__metrics">
          <span>会话 {{ sessions.length }}</span>
          <span>资料源 {{ sourceCount }}</span>
          <span>{{ todayLabel }}</span>
        </div>
        <button
          type="button"
          class="masthead__toggle"
          :aria-expanded="rightOpen"
          @click="rightOpen = !rightOpen"
        >
          {{ rightOpen ? '收起参数' : '展开参数' }}
        </button>
      </div>
    </header>

    <div class="workspace" :class="{ 'workspace--rail-collapsed': !rightOpen }">
      <aside class="west-rail" aria-label="模式与会话侧栏">
        <section class="rail-section">
          <header class="rail-section__head">
            <p>模式选择</p>
            <span>{{ modeMeta.eyebrow }}</span>
          </header>
          <div class="mode-stack" role="tablist" aria-label="模式切换">
            <button
              v-for="item in modeOptions"
              :key="item.value"
              type="button"
              class="mode-stack__item"
              :class="{ active: item.value === mode }"
              :aria-selected="item.value === mode"
              @click="mode = item.value"
            >
              <span class="mode-stack__label">{{ item.label }}</span>
              <span class="mode-stack__note">{{ item.note }}</span>
            </button>
          </div>
        </section>

        <section class="rail-section">
          <header class="rail-section__head rail-section__head--split">
            <div>
              <p>会话序列</p>
              <span>当前模式下继续追问也不会丢上下文</span>
            </div>
            <div class="rail-section__actions">
              <button type="button" class="text-button" @click="handleCreateSession">新建</button>
              <button type="button" class="text-button" @click="handleClearSession">清空</button>
            </div>
          </header>

          <ol class="session-list">
            <li v-for="session in sessions" :key="session.id">
              <button
                type="button"
                class="session-list__item"
                :class="{ active: session.id === activeSessionId }"
                :aria-pressed="session.id === activeSessionId"
                @click="handleSwitchSession(session.id)"
              >
                <span class="session-list__title">{{ session.title }}</span>
                <span class="session-list__time">{{ formatTime(session.updatedAt) }}</span>
                <span class="session-list__summary">
                  {{ session.isStreaming ? '正在生成回答...' : getSessionSummary(session) }}
                </span>
              </button>
            </li>
          </ol>
        </section>

        <section class="rail-section rail-section--quiet">
          <header class="rail-section__head">
            <p>Prompt Seeds</p>
            <span>给操作老师一个顺手的起手式</span>
          </header>
          <ul class="seed-list">
            <li v-for="item in featuredPrompts" :key="item.label">
              <button type="button" class="seed-list__item" @click="fillPrompt(item.prompt)">
                <span>{{ item.label }}</span>
                <small>{{ item.prompt }}</small>
              </button>
            </li>
          </ul>
        </section>
      </aside>

      <main class="stage">
        <section class="hero">
          <div class="hero__eyebrow">{{ modeMeta.eyebrow }}</div>
          <div class="briefing-strip" aria-label="当前模式摘要">
            <div v-for="item in briefingItems" :key="item.key" class="briefing-strip__item">
              <span>{{ item.key }}</span>
              <strong>{{ item.value }}</strong>
            </div>
          </div>
          <div class="hero__grid">
            <div class="hero__lead">
              <h2>{{ modeMeta.title }}</h2>
              <p>{{ modeMeta.note }}</p>
            </div>
            <dl class="hero__facts">
              <div>
                <dt>当前模式</dt>
                <dd>{{ modeMeta.label }}</dd>
              </div>
              <div>
                <dt>运行状态</dt>
                <dd>{{ streaming ? '回答生成中' : healthStateLabel }}</dd>
              </div>
              <div>
                <dt>当前模型</dt>
                <dd>{{ model }}</dd>
              </div>
            </dl>
          </div>
          <div v-if="pageNotice" class="hero__notice" :class="pageNotice.type">
            <span>{{ pageNotice.message }}</span>
            <button type="button" @click="pageNotice = null">关闭</button>
          </div>
        </section>

        <section v-if="mode === 'agent'" class="trace-band" aria-label="执行轨迹">
          <header class="trace-band__head">
            <div>
              <p>{{ traceBandTitle }}</p>
              <small>{{ traceBandNote }}</small>
            </div>
            <span>{{ visibleAgentTrace.length }} 步</span>
          </header>
          <div v-if="visibleAgentTrace.length" class="trace-band__list">
            <div v-for="item in visibleAgentTrace" :key="item.id" class="trace-band__item">
              <strong>{{ item.title }}</strong>
              <span class="trace-band__status" :class="item.status">{{ item.status }}</span>
              <span v-if="item.plan_step_index !== undefined">步骤 {{ item.plan_step_index }}</span>
              <span v-if="item.attempt !== undefined">尝试 {{ item.attempt }}</span>
              <span v-if="item.message">{{ item.message }}</span>
            </div>
          </div>
          <div v-else class="trace-band__empty">专家模式的轨迹会显示在这里。发起一次专家模式提问后，就能看到系统如何拆解和推进步骤。</div>
        </section>

        <section ref="messagesRef" class="transcript" aria-live="polite">
          <div v-if="hasOnlyWelcomeMessage" class="transcript__prologue">
            <p>开场建议</p>
            <h3>先拿一个明确问题把系统唤醒，再看它怎么组织信息、资料和回答口径。</h3>
            <div class="transcript__prologue-line">
              <span>建议从专业对比、录取规则、学院特色、报考建议四类问题起手。</span>
            </div>
          </div>

          <article
            v-for="message in messages"
            :key="message.id"
            class="entry"
            :class="[`entry--${message.role}`, { 'entry--degraded': message.status === 'degraded', 'entry--failed': message.status === 'failed' }]"
          >
            <header class="entry__meta">
              <span class="entry__role">{{ roleLabel(message.role) }}</span>
              <time>{{ formatTime(message.createdAt) }}</time>
            </header>

            <div v-if="message.status === 'failed'" class="entry__flag entry__flag--error">
              执行失败：{{ message.errorMessage || '本轮回答失败。' }}
            </div>
            <div
              v-else-if="message.status === 'degraded' && message.degradedFeatures?.length"
              class="entry__flag entry__flag--warning"
            >
              部分能力降级：{{ message.degradedFeatures.join(' / ') }}
            </div>

            <div class="entry__body">
              <MarkdownContent :content="message.content" />
            </div>

            <div v-if="message.sources?.length" class="entry__sources">
              <div v-for="source in message.sources" :key="source.url" class="entry__source">
                <span>{{ source.title }}</span>
                <span v-if="isLocalKnowledgeSource(source)">本地知识库</span>
                <a v-else :href="source.url" target="_blank" rel="noreferrer">查看来源</a>
              </div>
            </div>

            <details
              v-if="message.traceId || message.errorMessage || message.toolAudit?.length"
              class="entry__details"
            >
              <summary>{{ message.status === 'failed' ? '失败详情' : '执行详情' }}</summary>
              <div v-if="message.traceId" class="detail-row">
                <strong>Trace</strong>
                <code>{{ message.traceId }}</code>
              </div>
              <div v-if="message.errorMessage" class="detail-row">
                <strong>错误</strong>
                <span>{{ message.errorMessage }}</span>
              </div>
              <div v-if="message.toolAudit?.length" class="detail-stack">
                <strong>工具审计</strong>
                <code v-for="item in message.toolAudit" :key="item">{{ item }}</code>
              </div>
            </details>

            <details v-if="message.agentTrace?.length" class="entry__details">
              <summary>专家轨迹</summary>
              <div class="detail-stack">
                <div v-for="item in message.agentTrace" :key="item.id" class="detail-row detail-row--trace">
                  <strong>{{ item.title }}</strong>
                  <span>{{ item.status }}</span>
                  <span v-if="item.message">{{ item.message }}</span>
                </div>
              </div>
            </details>
          </article>

          <article v-if="streaming" class="entry entry--assistant entry--streaming">
            <header class="entry__meta">
              <span class="entry__role">系统回答</span>
              <time>{{ streamingText ? '流式输出中' : '正在准备回答' }}</time>
            </header>
            <div class="entry__body">
              <div v-if="!streamingText" class="loading-line">
                <span class="loading-dot"></span>
                <span>{{ mode === 'agent' ? '正在拆解问题并组织回答...' : mode === 'rag' ? '正在检索资料并生成回答...' : '正在生成回答...' }}</span>
              </div>
              <template v-else>
                <MarkdownContent :content="streamingText" />
                <span class="caret">▍</span>
              </template>
            </div>
          </article>
        </section>

        <form class="composer" @submit.prevent="handleSend">
          <div class="composer__header">
            <div>
              <p class="composer__eyebrow">Editorial Prompt Deck</p>
              <h3>把问题说具体，系统会在后面把检索、上下文和专家步骤兜住。</h3>
            </div>
            <span class="composer__hint">
              {{ blockedReason || (mode === 'chat' ? 'Enter 发送，Shift + Enter 换行' : '当前模式会自动追加检索与引用约束') }}
            </span>
          </div>

          <div class="composer__surface">
            <textarea
              ref="textareaRef"
              class="composer__input"
              :value="input"
              placeholder="例如：请结合 2025 招生章程和学院介绍，比较人工智能、自动化与电气工程的培养重点和适合人群。"
              aria-label="输入招生咨询问题"
              rows="4"
              @input="input = ($event.target as HTMLTextAreaElement).value"
              @keydown="handleComposerKeydown"
            ></textarea>

            <div class="composer__footer">
              <div class="composer__quick">
                <button type="button" class="ghost-button" @click="fillPrompt(featuredPrompts[0]?.prompt || '')">填入示例</button>
                <span>温度 {{ temperature.toFixed(2) }} · Top_p {{ topP.toFixed(2) }} · {{ agentStrategy === 'quality' ? '质量优先' : '速度优先' }}</span>
              </div>
              <div class="composer__actions">
                <button v-if="streaming" type="button" class="ghost-button" @click="handleStop">停止</button>
                <button type="submit" class="primary-button" :disabled="streaming || !canSend">发送问题</button>
              </div>
            </div>
          </div>
        </form>
      </main>

      <aside class="east-rail" :class="{ 'east-rail--closed': !rightOpen }" aria-label="参数与系统态势">
        <template v-if="rightOpen">
          <section class="rail-section">
            <header class="rail-section__head">
              <p>模型调优</p>
              <span>别把参数区做成会计报表</span>
            </header>

            <label class="field">
              <span>模型选择</span>
              <select :value="model" @change="model = ($event.target as HTMLSelectElement).value">
                <option v-for="item in modelOptions" :key="item" :value="item">{{ item }}</option>
              </select>
            </label>

            <label class="field">
              <span>Temperature <strong>{{ temperature.toFixed(2) }}</strong></span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                :value="temperature"
                @input="temperature = Number(($event.target as HTMLInputElement).value)"
              />
            </label>

            <label class="field">
              <span>Top_p <strong>{{ topP.toFixed(2) }}</strong></span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                :value="topP"
                @input="topP = Number(($event.target as HTMLInputElement).value)"
              />
            </label>

            <div v-if="mode === 'agent'" class="strategy-strip">
              <button
                type="button"
                class="strategy-strip__item"
                :class="{ active: agentStrategy === 'speed' }"
                :aria-pressed="agentStrategy === 'speed'"
                @click="agentStrategy = 'speed'"
              >
                速度优先
              </button>
              <button
                type="button"
                class="strategy-strip__item"
                :class="{ active: agentStrategy === 'quality' }"
                :aria-pressed="agentStrategy === 'quality'"
                @click="agentStrategy = 'quality'"
              >
                质量优先
              </button>
            </div>
          </section>

          <section class="rail-section">
            <header class="rail-section__head">
              <p>运行态势</p>
              <span>{{ healthStateLabel }}</span>
            </header>
            <div class="health-headline">
              <strong>{{ healthApp || 'gateway' }}</strong>
              <span class="health-badge" :class="{ ok: healthOverall, bad: !healthOverall }">
                {{ healthOverall ? '健康' : '异常' }}
              </span>
            </div>
            <ul class="health-list">
              <li v-for="dep in healthDependencies" :key="dep.name" class="health-list__item">
                <div>
                  <strong>{{ dep.name }}</strong>
                  <small v-if="dep.last_error">{{ dep.last_error }}</small>
                </div>
                <span :class="['health-badge', dep.healthy && !dep.circuit_open ? 'ok' : 'bad']">
                  {{ dep.healthy && !dep.circuit_open ? 'ok' : 'degraded' }}
                </span>
              </li>
            </ul>
          </section>

          <section class="rail-section">
            <header class="rail-section__head">
              <p>系统动作</p>
              <span>必要功能保留，不做花里胡哨的假控台</span>
            </header>
            <div class="ops-list">
              <button type="button" class="ghost-button" :disabled="healthLoading" @click="refreshHealth">
                {{ healthLoading ? '刷新中...' : '刷新健康状态' }}
              </button>
              <button type="button" class="ghost-button" :disabled="reindexLoading" @click="triggerReindex">
                {{ reindexLoading ? '执行中...' : '重建索引' }}
              </button>
              <button
                type="button"
                class="primary-button primary-button--muted"
                :disabled="compressLoading || !canCompressContext"
                @click="handleCompressContext"
              >
                {{ compressLoading ? '压缩中...' : '压缩当前上下文' }}
              </button>
            </div>
            <p v-if="reindexInfo" class="system-note">{{ reindexInfo }}</p>
            <p v-if="compressInfo" class="system-note">{{ compressInfo }}</p>
          </section>
        </template>

        <button v-else type="button" class="rail-reveal" :aria-expanded="rightOpen" @click="rightOpen = true">参数</button>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.shell {
  position: relative;
  min-height: 100vh;
  padding: 1.5rem;
  overflow: hidden;
}

.shell__wash,
.shell__grain {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.shell__wash {
  background:
    radial-gradient(circle at 12% 18%, rgba(202, 138, 4, 0.12), transparent 22%),
    radial-gradient(circle at 86% 12%, rgba(41, 37, 36, 0.08), transparent 26%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.52), transparent 48%);
}

.shell__grain {
  opacity: 0.22;
  background-image: linear-gradient(rgba(28, 25, 23, 0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(28, 25, 23, 0.02) 1px, transparent 1px);
  background-size: 22px 22px;
}

.masthead,
.west-rail,
.stage,
.east-rail {
  position: relative;
  z-index: 1;
}

.masthead {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 2rem;
  align-items: end;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid rgba(28, 25, 23, 0.12);
}

.masthead__eyebrow {
  margin: 0 0 0.75rem;
  font-size: 0.72rem;
  letter-spacing: 0.34em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.masthead h1 {
  margin: 0;
  font-family: var(--font-display);
  font-size: clamp(2.8rem, 6vw, 4.6rem);
  line-height: 0.94;
  letter-spacing: -0.03em;
  max-width: 11ch;
}

.masthead__deck {
  margin: 1rem 0 0;
  max-width: 44rem;
  font-size: 1rem;
  line-height: 1.65;
  color: var(--ink-2);
}

.masthead__aside {
  display: grid;
  gap: 0.85rem;
  justify-items: end;
}

.masthead__metrics {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.85rem 1.25rem;
  font-size: 0.78rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.masthead__toggle {
  padding: 0.72rem 1.1rem;
  border: 1px solid rgba(28, 25, 23, 0.14);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.7);
  color: var(--ink-1);
  transition: border-color 0.2s ease, transform 0.2s ease;
}

.masthead__toggle:hover {
  transform: translateY(-1px);
  border-color: rgba(202, 138, 4, 0.34);
}

.workspace {
  display: grid;
  grid-template-columns: minmax(15rem, 17rem) minmax(0, 1fr) minmax(18rem, 20rem);
  gap: 2rem;
  padding-top: 1.75rem;
  min-height: calc(100vh - 13rem);
}

.workspace--rail-collapsed {
  grid-template-columns: minmax(15rem, 17rem) minmax(0, 1fr) 4rem;
}

.west-rail,
.east-rail {
  display: grid;
  align-content: start;
  gap: 1.5rem;
}

.stage {
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr) auto;
  gap: 1.35rem;
  min-width: 0;
}

.rail-section {
  padding-top: 1rem;
  border-top: 1px solid rgba(28, 25, 23, 0.1);
}

.rail-section:first-child {
  padding-top: 0;
  border-top: none;
}

.rail-section__head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1rem;
}

.rail-section__head--split {
  align-items: flex-start;
}

.rail-section__head p {
  margin: 0;
  font-size: 0.72rem;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.rail-section__head span {
  font-size: 0.82rem;
  line-height: 1.5;
  color: var(--ink-2);
}

.rail-section__actions {
  display: flex;
  gap: 0.8rem;
}

.text-button {
  border: none;
  padding: 0;
  background: transparent;
  color: var(--accent);
  font-size: 0.85rem;
}

.mode-stack {
  display: grid;
  gap: 0.6rem;
}

.mode-stack__item {
  display: grid;
  gap: 0.4rem;
  padding: 0.8rem 0;
  border: none;
  border-bottom: 1px solid rgba(28, 25, 23, 0.08);
  background: transparent;
  text-align: left;
  color: var(--ink-1);
  transition: color 0.2s ease, transform 0.2s ease;
}

.mode-stack__item:hover {
  transform: translateX(4px);
}

.mode-stack__item.active {
  color: var(--accent);
}

.mode-stack__label {
  font-size: 1rem;
  font-weight: 600;
}

.mode-stack__note {
  font-size: 0.84rem;
  line-height: 1.55;
  color: var(--ink-2);
}

.session-list,
.seed-list,
.health-list {
  display: grid;
  gap: 0.5rem;
  list-style: none;
  margin: 0;
  padding: 0;
}

.session-list__item,
.seed-list__item {
  width: 100%;
  display: grid;
  gap: 0.3rem;
  padding: 0.7rem 0;
  border: none;
  border-bottom: 1px solid rgba(28, 25, 23, 0.08);
  background: transparent;
  text-align: left;
  transition: transform 0.2s ease, color 0.2s ease;
}

.session-list__item:hover,
.seed-list__item:hover {
  transform: translateX(4px);
}

.session-list__item.active {
  color: var(--accent);
}

.session-list__title {
  font-weight: 600;
}

.session-list__time,
.session-list__summary,
.seed-list__item small {
  color: var(--ink-2);
  font-size: 0.82rem;
  line-height: 1.5;
}

.rail-section--quiet {
  opacity: 0.92;
}

.hero {
  display: grid;
  gap: 1rem;
  padding-bottom: 1.3rem;
  border-bottom: 1px solid rgba(28, 25, 23, 0.12);
}

.briefing-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid rgba(28, 25, 23, 0.08);
}

.briefing-strip__item {
  display: grid;
  gap: 0.35rem;
}

.briefing-strip__item span {
  font-size: 0.68rem;
  letter-spacing: 0.26em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.briefing-strip__item strong {
  font-size: 0.92rem;
  line-height: 1.55;
  color: var(--ink-1);
}

.hero__eyebrow,
.composer__eyebrow {
  margin: 0;
  font-size: 0.72rem;
  letter-spacing: 0.34em;
  text-transform: uppercase;
  color: var(--accent);
}

.hero__grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(14rem, 17rem);
  gap: 2rem;
  align-items: start;
}

.hero__lead h2 {
  margin: 0;
  font-family: var(--font-display);
  font-size: clamp(2.2rem, 4vw, 3.6rem);
  line-height: 0.98;
  letter-spacing: -0.03em;
}

.hero__lead p {
  margin: 0.85rem 0 0;
  max-width: 38rem;
  font-size: 0.98rem;
  line-height: 1.68;
  color: var(--ink-2);
}

.hero__facts {
  display: grid;
  gap: 0.8rem;
  margin: 0;
}

.hero__facts div {
  display: grid;
  gap: 0.18rem;
  padding: 0.45rem 0;
  border-bottom: 1px solid rgba(28, 25, 23, 0.08);
}

.hero__facts dt {
  font-size: 0.7rem;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.hero__facts dd {
  margin: 0;
  font-size: 1rem;
  color: var(--ink-1);
}

.hero__notice {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  padding-top: 0.85rem;
  border-top: 1px solid rgba(28, 25, 23, 0.08);
  font-size: 0.92rem;
  line-height: 1.6;
}

.hero__notice button {
  border: none;
  padding: 0;
  background: transparent;
  color: inherit;
}

.hero__notice.info {
  color: var(--info);
}

.hero__notice.warning {
  color: var(--warning);
}

.hero__notice.error {
  color: var(--danger);
}

.trace-band {
  display: grid;
  gap: 0.85rem;
  padding-bottom: 1.2rem;
  border-bottom: 1px solid rgba(28, 25, 23, 0.12);
}

.trace-band__head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: flex-start;
  color: var(--ink-3);
}

.trace-band__head p {
  margin: 0;
  font-size: 0.78rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.trace-band__head small {
  display: block;
  margin-top: 0.35rem;
  font-size: 0.82rem;
  line-height: 1.5;
  color: var(--ink-2);
}

.trace-band__list {
  display: grid;
  gap: 0.55rem;
}

.trace-band__empty {
  padding: 0.85rem 0;
  border-top: 1px solid rgba(28, 25, 23, 0.08);
  font-size: 0.9rem;
  line-height: 1.65;
  color: var(--ink-2);
}

.trace-band__item {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem 0.9rem;
  padding: 0.7rem 0;
  border-bottom: 1px solid rgba(28, 25, 23, 0.08);
  font-size: 0.88rem;
  line-height: 1.55;
}

.trace-band__status.completed {
  color: var(--success);
}

.trace-band__status.retrying,
.trace-band__status.degraded,
.trace-band__status.failed {
  color: var(--danger);
}

.transcript {
  min-height: 0;
  overflow-y: auto;
  padding-right: 0.4rem;
}

.transcript__prologue {
  display: grid;
  gap: 0.65rem;
  width: min(100%, 48rem);
  padding: 0 0 1.6rem;
  border-bottom: 1px solid rgba(28, 25, 23, 0.08);
  margin-bottom: 1rem;
}

.transcript__prologue p {
  margin: 0;
  font-size: 0.72rem;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: var(--accent);
}

.transcript__prologue h3 {
  margin: 0;
  font-family: var(--font-display);
  font-size: clamp(1.65rem, 2.4vw, 2.3rem);
  line-height: 1.08;
  letter-spacing: -0.02em;
}

.transcript__prologue-line {
  padding-top: 0.55rem;
  border-top: 1px solid rgba(28, 25, 23, 0.08);
  font-size: 0.9rem;
  line-height: 1.65;
  color: var(--ink-2);
}

.entry {
  width: min(100%, 56rem);
  padding: 1.15rem 0;
  border-top: 1px solid rgba(28, 25, 23, 0.08);
  animation: entry-rise 0.4s ease both;
  --markdown-ink: var(--ink-1);
  --markdown-link: var(--accent);
  --markdown-code-bg: rgba(202, 138, 4, 0.1);
  --markdown-pre-bg: rgba(28, 25, 23, 0.94);
  --markdown-pre-ink: #f8f7f4;
  --markdown-mermaid-bg: rgba(255, 255, 255, 0.78);
  --markdown-mermaid-border: rgba(28, 25, 23, 0.08);
}

.entry:first-child {
  border-top: none;
  padding-top: 0;
}

.entry--user {
  margin-left: auto;
  width: min(78%, 44rem);
  padding-left: 1.35rem;
  border-left: 2px solid rgba(202, 138, 4, 0.46);
  background: linear-gradient(90deg, rgba(202, 138, 4, 0.08), transparent 72%);
  --markdown-ink: var(--ink-0);
}

.entry--system {
  color: var(--ink-2);
}

.entry__meta {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.8rem;
  font-size: 0.74rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.entry__role {
  font-weight: 600;
}

.entry__body {
  font-size: 0.98rem;
  line-height: 1.75;
}

.entry__flag {
  margin-bottom: 0.75rem;
  font-size: 0.85rem;
  line-height: 1.55;
}

.entry__flag--warning {
  color: var(--warning);
}

.entry__flag--error {
  color: var(--danger);
}

.entry__sources {
  display: grid;
  gap: 0.55rem;
  margin-top: 1rem;
}

.entry__source {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 0.6rem;
  padding-top: 0.6rem;
  border-top: 1px dashed rgba(28, 25, 23, 0.08);
  font-size: 0.82rem;
  color: var(--ink-2);
}

.entry__source a {
  color: var(--accent);
  text-decoration: none;
}

.entry__details {
  margin-top: 0.9rem;
  padding-top: 0.8rem;
  border-top: 1px solid rgba(28, 25, 23, 0.08);
}

.entry__details summary {
  cursor: pointer;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--accent);
}

.detail-row,
.detail-stack {
  display: grid;
  gap: 0.5rem;
  margin-top: 0.75rem;
  font-size: 0.84rem;
  line-height: 1.6;
}

.detail-row code,
.detail-stack code {
  padding: 0.55rem 0.7rem;
  border-radius: 0.55rem;
  background: rgba(28, 25, 23, 0.05);
  overflow-wrap: anywhere;
}

.detail-row--trace {
  padding-bottom: 0.65rem;
  border-bottom: 1px solid rgba(28, 25, 23, 0.06);
}

.entry--streaming {
  color: var(--ink-1);
}

.loading-line {
  display: inline-flex;
  align-items: center;
  gap: 0.7rem;
}

.loading-dot {
  width: 0.72rem;
  height: 0.72rem;
  border-radius: 999px;
  background: var(--accent);
  animation: pulse 1.2s ease-in-out infinite;
}

.caret {
  animation: caret-blink 0.9s step-start infinite;
}

.composer {
  padding-top: 1.2rem;
  border-top: 1px solid rgba(28, 25, 23, 0.12);
}

.composer__header {
  display: flex;
  justify-content: space-between;
  gap: 1.5rem;
  margin-bottom: 1rem;
}

.composer__header h3 {
  margin: 0.45rem 0 0;
  font-family: var(--font-display);
  font-size: clamp(1.4rem, 2vw, 2rem);
  line-height: 1.08;
}

.composer__hint {
  max-width: 18rem;
  font-size: 0.84rem;
  line-height: 1.6;
  color: var(--ink-2);
  text-align: right;
}

.composer__surface {
  display: grid;
  gap: 1rem;
  padding: 1rem 0 0;
  border-top: 1px solid rgba(28, 25, 23, 0.08);
}

.composer__input {
  width: 100%;
  min-height: 7.5rem;
  padding: 0;
  border: none;
  background: transparent;
  resize: vertical;
  color: var(--ink-0);
  font-size: 1rem;
  line-height: 1.8;
  outline: none;
}

.composer__footer {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
  padding-top: 0.8rem;
  border-top: 1px solid rgba(28, 25, 23, 0.08);
}

.composer__quick {
  display: flex;
  flex-wrap: wrap;
  gap: 0.9rem;
  align-items: center;
  font-size: 0.82rem;
  color: var(--ink-2);
}

.composer__actions,
.ops-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  justify-content: flex-end;
}

.ghost-button,
.primary-button,
.rail-reveal,
.strategy-strip__item,
.field select {
  font: inherit;
}

.ghost-button,
.primary-button,
.strategy-strip__item,
.rail-reveal {
  border-radius: 999px;
  padding: 0.78rem 1.15rem;
}

.ghost-button,
.strategy-strip__item,
.rail-reveal {
  border: 1px solid rgba(28, 25, 23, 0.12);
  background: rgba(255, 255, 255, 0.72);
  color: var(--ink-1);
}

.primary-button {
  border: 1px solid rgba(28, 25, 23, 0.08);
  background: linear-gradient(135deg, #d6a64a, #8e6817);
  color: #fffdf8;
  box-shadow: 0 12px 24px rgba(142, 104, 23, 0.16);
}

.primary-button--muted {
  background: linear-gradient(135deg, #2b2725, #1c1917);
  box-shadow: none;
}

.primary-button:disabled,
.ghost-button:disabled {
  opacity: 0.58;
  cursor: not-allowed;
}

.field {
  display: grid;
  gap: 0.75rem;
  margin-bottom: 1rem;
}

.field span {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  font-size: 0.84rem;
  color: var(--ink-2);
}

.field strong {
  color: var(--ink-1);
}

.field select {
  width: 100%;
  padding: 0.8rem 1rem;
  border: 1px solid rgba(28, 25, 23, 0.12);
  border-radius: 0.9rem;
  background: rgba(255, 255, 255, 0.78);
  color: var(--ink-0);
}

.field input[type='range'] {
  width: 100%;
  accent-color: var(--accent);
}

.strategy-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
}

.strategy-strip__item.active {
  border-color: rgba(202, 138, 4, 0.34);
  color: var(--accent);
}

.health-headline {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
  margin-bottom: 0.8rem;
}

.health-headline strong {
  font-size: 1rem;
}

.health-list__item {
  display: flex;
  justify-content: space-between;
  gap: 0.8rem;
  align-items: flex-start;
  padding: 0.65rem 0;
  border-bottom: 1px solid rgba(28, 25, 23, 0.08);
}

.health-list__item strong {
  display: block;
  margin-bottom: 0.2rem;
}

.health-list__item small {
  color: var(--ink-3);
  line-height: 1.5;
}

.health-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.3rem 0.6rem;
  border-radius: 999px;
  font-size: 0.74rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.health-badge.ok {
  background: rgba(22, 101, 52, 0.1);
  color: var(--success);
}

.health-badge.bad {
  background: rgba(185, 28, 28, 0.1);
  color: var(--danger);
}

.system-note {
  margin: 0.9rem 0 0;
  font-size: 0.82rem;
  line-height: 1.6;
  color: var(--ink-2);
}

.east-rail--closed {
  display: flex;
  align-items: start;
  justify-content: center;
}

.rail-reveal {
  margin-top: 0.25rem;
}

@media (max-width: 1280px) {
  .workspace {
    grid-template-columns: minmax(13rem, 15rem) minmax(0, 1fr) minmax(16rem, 18rem);
    gap: 1.5rem;
  }

  .briefing-strip {
    grid-template-columns: 1fr;
    gap: 0.8rem;
  }

  .hero__grid {
    grid-template-columns: 1fr;
    gap: 1.2rem;
  }
}

@media (max-width: 1080px) {
  .shell {
    padding: 1.1rem;
  }

  .masthead {
    grid-template-columns: 1fr;
    gap: 1.25rem;
  }

  .masthead__aside {
    justify-items: start;
  }

  .masthead__metrics {
    justify-content: flex-start;
  }

  .workspace,
  .workspace--rail-collapsed {
    grid-template-columns: 1fr;
  }

  .stage {
    min-height: 0;
  }

  .east-rail--closed {
    display: none;
  }
}

@media (max-width: 768px) {
  .shell {
    padding: 0.9rem;
  }

  .masthead h1 {
    max-width: none;
    font-size: clamp(2.35rem, 14vw, 3.35rem);
  }

  .entry--user {
    width: 100%;
  }

  .composer__header,
  .composer__footer {
    flex-direction: column;
    align-items: flex-start;
  }

  .composer__hint {
    max-width: none;
    text-align: left;
  }

  .composer__actions,
  .ops-list {
    width: 100%;
    justify-content: flex-start;
  }
}
</style>
