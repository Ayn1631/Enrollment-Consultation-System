import type { ChatMode } from './types'

export type SupportedMode = Extract<ChatMode, 'chat' | 'rag' | 'agent'>

export interface ModeDisplay {
  value: SupportedMode
  label: string
  eyebrow: string
  title: string
  description: string
  note: string
}

export interface PromptSeed {
  label: string
  prompt: string
}

export const modeOptions: ModeDisplay[] = [
  {
    value: 'chat',
    label: '对话模式',
    eyebrow: 'Direct Dialogue',
    title: '把复杂招生问题讲清楚，不靠花架子撑场面。',
    description: '适合连续咨询、政策解释和专业对比。系统会保持语气克制，优先把问题讲明白。',
    note: '自然问答优先，适合老师、考生和家长进行连续追问。'
  },
  {
    value: 'rag',
    label: 'RAG 问答',
    eyebrow: 'Retrieval Grounded',
    title: '把答案拴在资料上，少拍脑袋，多给依据。',
    description: '适合招生章程、学院介绍、时间节点和政策条款类问题，强调检索支撑与引用约束。',
    note: '检索与引用约束默认开启，适合需要稳健依据的官方口径场景。'
  },
  {
    value: 'agent',
    label: '专家模式',
    eyebrow: 'Expert Runtime',
    title: '让系统自己拆问题、跑步骤、给结果，不在前台卖弄流程。',
    description: '适合多条件志愿推荐、复杂政策判断和跨资料综合分析，展示必要轨迹，不把界面搞得像监控大屏。',
    note: '会启用检索、联网和工具链，适合复杂问题与高价值问答。'
  }
]

export const promptSeeds: Record<SupportedMode, PromptSeed[]> = {
  chat: [
    {
      label: '本科专业咨询',
      prompt: '河南考生 560 分，物化生组合，适合报考中原工学院哪些专业？'
    },
    {
      label: '家长问法',
      prompt: '孩子想学人工智能，但担心就业和数学基础，报考时应该怎么看专业差异？'
    },
    {
      label: '专业对比',
      prompt: '计算机学院、人工智能学院、网络空间安全学院的培养方向有什么区别？'
    }
  ],
  rag: [
    {
      label: '章程检索',
      prompt: '请依据 2025 年招生章程说明中原工学院的录取规则和体检要求。'
    },
    {
      label: '学院资料',
      prompt: '请结合学院介绍，概括软件学院的培养特色、核心课程和就业方向。'
    },
    {
      label: '时间节点',
      prompt: '校园开放日、招生咨询和报考资料获取渠道有哪些官方入口？'
    }
  ],
  agent: [
    {
      label: '志愿分析',
      prompt: '我在贵州高考 520 分，物化生组合，想报工科，请帮我拆解可报专业并给出建议。'
    },
    {
      label: '复杂筛选',
      prompt: '请综合录取规则、专业匹配和就业方向，帮我比较自动化、电气工程和人工智能这三个方向。'
    },
    {
      label: '答疑预案',
      prompt: '如果家长担心住宿、学费、转专业和就业，请给我一套结构清晰的现场答疑口径。'
    }
  ]
}

export const editorialBriefings: Record<SupportedMode, Array<{ key: string; value: string }>> = {
  chat: [
    { key: 'tone', value: '口径自然，但不轻浮' },
    { key: 'focus', value: '把专业、录取、就业差异讲透' },
    { key: 'scene', value: '适合现场咨询与连续追问' }
  ],
  rag: [
    { key: 'tone', value: '优先资料依据，压缩拍脑袋空间' },
    { key: 'focus', value: '招生章程、学院资料、时间节点' },
    { key: 'scene', value: '适合官方口径与政策核对' }
  ],
  agent: [
    { key: 'tone', value: '前台克制，后台拆解' },
    { key: 'focus', value: '复杂问题分步分析与综合建议' },
    { key: 'scene', value: '适合高价值问答与多条件筛选' }
  ]
}
