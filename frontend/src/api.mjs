import { anonymousReportBlob, configuredPresentationAliases, createPresentationPolicy, presentationEnabled } from './presentation.mjs'

export function createApiClient({ getToken = () => '', onUnauthorized = () => {}, fetchImpl = fetch, presentation = presentationEnabled, aliases } = {}) {
  const policy = presentation ? createPresentationPolicy(aliases ?? configuredPresentationAliases()) : null
  return async (path, { method = 'GET', body, signal, responseType } = {}) => {
    const anonymousExport = Boolean(policy && responseType === 'blob' && /^\/evaluations\/[^/]+\/export(?:\?|$)/.test(path))
    if (policy && responseType === 'blob' && !anonymousExport) throw new Error('展示副本提供资料摘录，请从资料页下载摘录。原件保存在本机资料库。')
    const headers = new Headers({ Accept: 'application/json' })
    const token = getToken()
    if (token) headers.set('Authorization', `Bearer ${token}`)
    const form = typeof FormData !== 'undefined' && body instanceof FormData
    if (body !== undefined && !form) headers.set('Content-Type', 'application/json')
    const outgoing = policy && !path.startsWith('/auth/') ? (form ? policy.restoreForm(body) : policy.restore(body)) : body
    let response
    try {
      response = await fetchImpl(`/api${policy ? policy.restorePath(path) : path}`, {
        method, headers, signal, body: outgoing === undefined ? undefined : form ? outgoing : JSON.stringify(outgoing),
      })
    } catch (error) {
      if (error.name === 'AbortError') throw error
      throw new Error('后端连接失败，请检查本地服务是否启动。')
    }
    if (response.ok && responseType === 'blob' && !anonymousExport) return response.blob()
    const raw = await response.text()
    let data
    try { data = raw ? JSON.parse(raw) : null } catch { data = null }
    if (policy) data = policy.redact(data)
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
    if (anonymousExport) {
      if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('评估导出内容无效。')
      return anonymousReportBlob(data)
    }
    return data
  }
}
