export function createApiClient({ getToken = () => '', onUnauthorized = () => {}, fetchImpl = fetch } = {}) {
  return async (path, { method = 'GET', body, signal } = {}) => {
    const headers = new Headers({ Accept: 'application/json' })
    const token = getToken()
    if (token) headers.set('Authorization', `Bearer ${token}`)
    const form = typeof FormData !== 'undefined' && body instanceof FormData
    if (body !== undefined && !form) headers.set('Content-Type', 'application/json')
    let response
    try {
      response = await fetchImpl(`/api${path}`, {
        method, headers, signal, body: body === undefined ? undefined : form ? body : JSON.stringify(body),
      })
    } catch (error) {
      if (error.name === 'AbortError') throw error
      throw new Error('后端连接失败，请检查本地服务是否启动。')
    }
    const raw = await response.text()
    let data
    try { data = raw ? JSON.parse(raw) : null } catch { data = null }
    if (!response.ok) {
      if (response.status === 401) onUnauthorized()
      const detail = data?.message || data?.detail
      const message = Array.isArray(detail)
        ? detail.map(item => `${item.loc?.join('.') || ''} ${item.msg || ''}`).join('；')
        : typeof detail === 'string' ? detail : detail?.message
      const error = new Error(message || `请求未完成（${response.status}）`)
      error.status = response.status
      throw error
    }
    return data
  }
}
