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
})
