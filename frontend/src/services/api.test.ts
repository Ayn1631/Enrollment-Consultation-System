import { afterEach, describe, expect, test, vi } from 'vitest'

type FetchResponse = {
  ok: boolean
  status: number
  json: () => Promise<unknown>
}

async function loadApiModule(token: string): Promise<typeof import('./api')> {
  vi.resetModules()
  vi.stubEnv('VITE_API_BASE_URL', 'http://example.com')
  vi.stubEnv('VITE_ADMIN_API_TOKEN', token)
  return import('./api')
}

function mockFetch(payload: unknown) {
  const mocked = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => payload
  } as FetchResponse)
  vi.stubGlobal('fetch', mocked)
  return mocked
}

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('api.postReindex', () => {
  test('配置管理员令牌时自动附带请求头', async () => {
    const fetchMock = mockFetch({ status: 'ok', result: { chunks: 8 } })
    const api = await loadApiModule('admin-secret')
    await api.postReindex()

    const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe('http://example.com/api/admin/reindex')
    expect(options.method).toBe('POST')
    expect(options.headers).toEqual({ 'x-admin-token': 'admin-secret' })
  })

  test('未配置管理员令牌时不携带鉴权头', async () => {
    const fetchMock = mockFetch({ status: 'ok', result: { chunks: 8 } })
    const api = await loadApiModule('')
    await api.postReindex()

    const [, options] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(options.headers).toEqual({})
  })

  test('后端返回结构化错误时保留详细提示和 trace_id', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({
          detail: {
            message: '后端处理聊天请求时发生异常，请查看后端日志。',
            trace_id: 'trace-123'
          }
        })
      } as FetchResponse)
    )

    const api = await loadApiModule('')
    await expect(api.postReindex()).rejects.toMatchObject({
      name: 'ApiRequestError',
      message: '后端处理聊天请求时发生异常，请查看后端日志。（trace_id: trace-123）',
      status: 500,
      traceId: 'trace-123'
    })
  })
})

describe('api.startChatStream', () => {
  test('可以解析 step、message、done 三类 SSE 事件', async () => {
    const encoder = new TextEncoder()
    const streamBody = new ReadableStream({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            [
              'event: step',
              'data: {"id":"step-1","node":"load_memory","title":"加载会话记忆","status":"completed","strategy":"quality","timestamp":"2026-03-26T00:00:00Z"}',
              '',
              'event: message',
              'data: {"delta":"最终回答片段"}',
              '',
              'event: done',
              'data: {"finish_reason":"stop","status":"failed","agent_strategy":"quality","trace_id":"trace-001","error_message":"generation failure injected","tool_audit":["agent:error:RuntimeError"]}',
              ''
            ].join('\n')
          )
        )
        controller.close()
      }
    })

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: streamBody,
        json: async () => ({})
      } as Response)
    )

    const api = await loadApiModule('')
    const deltas: string[] = []
    const steps: Array<Record<string, unknown>> = []
    const doneEvents: Array<Record<string, unknown>> = []
    const errors: Error[] = []

    await api.startChatStream(
      {
        session_id: 'session-1',
        messages: [{ role: 'user', content: '你好' }],
        features: ['rag'],
        mode: 'agent',
        stream: true,
        agent_strategy: 'quality'
      },
      {
        onDelta: (delta) => deltas.push(delta),
        onStep: (event) => steps.push(event as unknown as Record<string, unknown>),
        onDone: (event) => doneEvents.push(event as Record<string, unknown>),
        onError: (error) => errors.push(error)
      }
    )

    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(errors).toEqual([])
    expect(deltas).toEqual(['最终回答片段'])
    expect(steps[0]?.node).toBe('load_memory')
    expect(doneEvents[0]?.agent_strategy).toBe('quality')
    expect(doneEvents[0]?.trace_id).toBe('trace-001')
    expect(doneEvents[0]?.error_message).toBe('generation failure injected')
  })
})
