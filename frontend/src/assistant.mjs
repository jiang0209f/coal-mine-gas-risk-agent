import { referenceLocation } from './knowledge.mjs'
import { mineTime } from './utils.mjs'

export function chatDeepLink(search = '') {
  const params = new URLSearchParams(search)
  const session_id = params.get('session_id')?.trim()
  return params.get('view') === 'chat' && session_id ? { view: 'chat', session_id } : null
}

export function assistantChannel(task) {
  const method = task?.method || task?.brief_method
  if (method === 'deepseek' && typeof task?.model === 'string' && task.model.trim() && task?.chat_scope === 'general') return 'DeepSeek通用对话'
  if (method === 'structured_template' || task?.error_code === 'cloud_source_not_authorized'
      || task?.local_reason === 'cloud_source_not_authorized') return '本机查询'
  return method === 'deepseek' && typeof task?.model === 'string' && task.model.trim() ? 'DeepSeek' : '业务查询'
}

export function chatStatusView(status) {
  if (!status) return { title: '等待选择会话', detail: '选择会话后显示当前来源的问答接入状态。', tone: 'muted', switchSession: null }
  if (!status.api_configured || status.mode === 'not_configured') {
    return { title: 'DeepSeek未配置', detail: '当前页面尚不能调用DeepSeek API。', tone: 'warning', switchSession: null }
  }
  if (status.mode === 'summary' || status.mode === 'synthetic') {
    return { title: status.mode === 'summary' ? '当前来源已接入DeepSeek' : '当前演练已接入DeepSeek',
      detail: '业务问题由DeepSeek调用工具处理，实际调用过程随回答展示。', tone: 'connected', switchSession: null }
  }
  const sessions = (Array.isArray(status.available_sessions) ? status.available_sessions : [])
    .filter(item => typeof item?.session_id === 'string' && item.session_id && typeof item.source_id === 'string' && item.source_id)
  return { title: status.general_chat ? '通用对话可用' : '当前来源未接入DeepSeek',
    detail: '当前来源未启用API数据分析；请切换到已授权回放后查询业务数据。', tone: 'warning',
    switchSession: sessions.find(item => item.source_id.endsWith('-t0')) || sessions[0] || null }
}

export function assistantMessage(task, states = {}) {
  return { role: 'assistant', text: task.final_text || task.answer || task.message || `任务状态：${states[task.state] || task.state}`,
    state: task.state, task_id: task.task_id, method: task.method || task.brief_method, model: task.model,
    steps: task.tool_steps, local_reason: task.local_reason, error_code: task.error_code,
    chat_scope: task.chat_scope, chat_status: task.chat_status }
}

const tools = { query_series: '查询监测序列', run_prediction: '运行预测', get_forecast: '读取预测结果', query_events: '查询预警事件', get_event_evidence: '读取事件依据', search_documents: '检索批准资料', draft_advice: '生成建议草稿', get_case: '查询处置进展' }
const states = { queued: '排队中', started: '执行中', running: '执行中', success: '已完成', completed: '已完成', error: '失败', failed: '失败', cancelled: '已取消' }
export const toolName = name => tools[name] || '业务操作'
export const toolState = state => states[state] || '待确认'
export function toolArguments(value) {
  const labels = { event_id: '事件', event_revision: '修订', point_id: '测点', run_id: '预测运行', query: '关键词', start_at: '开始', end_at: '结束' }
  return Object.entries(value || {}).filter(([key, item]) => labels[key] && ['string', 'number'].includes(typeof item))
    .map(([key, item]) => `${labels[key]}：${key.endsWith('_at') ? mineTime(item) : String(item).slice(0, 140)}`).join(' · ')
}

export async function prepareReferenceDownload({ presentation, document, file, api }) {
  if (!presentation) return { filename: file.filename,
    blob: await api(`/references/files/${encodeURIComponent(file.file_id)}/content`, { responseType: 'blob' }) }
  const fragments = (document?.fragments || []).filter(item => typeof item.text === 'string' && item.text.trim())
  if (!fragments.length) throw new Error('暂无可导出的资料摘录')
  const title = document.title || '资料'
  const filename = String(title).replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_') + '-摘录.txt'
  const parts = [title, document.version ? `版本：${document.version}` : '', document.source ? `来源：${document.source}` : '',
    ...fragments.map(item => `${referenceLocation(item.locator || {})}\n${item.text}`)].filter(Boolean)
  return { filename, blob: new Blob([parts.join('\n\n') + '\n'], { type: 'text/plain;charset=utf-8' }) }
}
