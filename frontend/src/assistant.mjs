import { referenceLocation } from './knowledge.mjs'
import { mineTime } from './utils.mjs'

export function assistantChannel(task) {
  const method = task?.method || task?.brief_method
  if (method === 'structured_template' || task?.error_code === 'cloud_source_not_authorized'
      || task?.local_reason === 'cloud_source_not_authorized') return '本机查询'
  return method === 'deepseek' && typeof task?.model === 'string' && task.model.trim() ? 'DeepSeek' : '业务查询'
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
