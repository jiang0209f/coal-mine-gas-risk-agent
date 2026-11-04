import { contextKey } from './workspace.mjs'

const businessLabels = {
  observed_exceedance: '实测超限', forecast_exceedance: '预测预警',
  availability_inferred: '可用时间推定', unit_inferred: '单位推定',
  suspected_calibration: '疑似标校', planned_calibration: '计划标校', confirmed_calibration: '已确认标校',
  invalid_value: '无效值', duplicate_timestamp: '重复时间', timezone_inferred: '时区按配置', forward_filled: '过去值填补',
}
const businessCode = new RegExp(`(?<![A-Za-z0-9_.:/-])(${Object.keys(businessLabels).join('|')})(?![A-Za-z0-9_.:/-])`, 'g')

export const briefScopeKey = (session, event) => `${contextKey(session, event?.point_id)}:${event?.event_id || ''}:${event?.event_revision || ''}`

export function briefMatches(brief, session, event) {
  if (!brief || !session || !event || brief.session_id !== session.session_id || brief.source_id !== session.source_id || brief.event_id !== event.event_id) return false
  const actual = Number(brief.event_revision), requested = Number(brief.requested_event_revision ?? brief.event_revision)
  if (requested !== Number(event.event_revision) || actual > requested || actual < 1 || (actual !== requested && !brief.stale)) return false
  if (brief.as_of && (!Number.isFinite(Date.parse(brief.as_of)) || Date.parse(brief.as_of) > Date.parse(session.as_of))) return false
  const task = brief.task
  return !task || (task.session_id === session.session_id && task.source_id === session.source_id && task.event_id === event.event_id && Number(task.event_revision) === actual)
}

export const shouldPollBrief = brief => ['queued', 'running'].includes(brief?.status)

export function briefCitations(brief) {
  const ids = new Set()
  return (brief?.task?.citations || []).map(item => item?.reference ? { ...item.reference, ...item } : item).filter(item => {
    if (!item || !/^[A-Z]\d+$/.test(item.citation_id) || ids.has(item.citation_id) || !['event', 'rule', 'observation', 'forecast_run', 'reference', 'attribution_run'].includes(item.kind)) return false
    ids.add(item.citation_id)
    return true
  })
}

export function briefTextParts(text, citations) {
  text = String(text || '').replace(/^ {0,3}#{1,6}[\t ]+/gm, '')
    .replace(/\*\*(?=\S)([^\n]*?\S)\*\*/g, '$1')
    .replace(/(?<![A-Za-z0-9_./:-])__(?=\S)([^\n]*?\S)__(?![A-Za-z0-9_./:-])/g, '$1')
    .replace(businessCode, (code, _matched, offset, source) => {
      const identifier = /\b(?:model_id|model_version|version|point_id|source_id|task_id|run_id)\s*[:=：]\s*[`"']?$/i.test(source.slice(0, offset))
      return identifier ? code : businessLabels[code]
    })
  const known = new Map(citations.map(item => [item.citation_id, item]))
  const parts = [], pattern = /\[([A-Z]\d+)\]/g
  let cursor = 0
  for (const match of text.matchAll(pattern)) {
    const citation = known.get(match[1])
    if (!citation) continue
    if (match.index > cursor) parts.push({ text: text.slice(cursor, match.index) })
    parts.push({ text: match[0], citation })
    cursor = match.index + match[0].length
  }
  if (cursor < text.length) parts.push({ text: text.slice(cursor) })
  return parts
}

export function localCitationDetail(brief, citation) {
  const cards = brief?.task?.result_cards || []
  const evidence = cards.find(card => card.tool === 'get_event_evidence')?.data
  const title = citation.label || '引用依据'
  if (citation.kind === 'event' && evidence?.event_snapshot) return { title, event: evidence.event_snapshot }
  if (citation.kind === 'rule' && evidence?.rules) return { title, event: evidence.event_snapshot, rules: evidence.rules }
  if (citation.kind === 'observation') {
    const observations = (evidence?.observation_fragments || []).filter(row => (row.id || row.observation_id) === citation.observation_id)
    if (observations.length) return { title, observations, summary: `来源文件 ${citation.source_file_id || observations[0].source_file_id} · 原始行 ${citation.row_number ?? observations[0].row_number}` }
  }
  if (citation.kind === 'forecast_run') {
    const forecast = [evidence?.forecast_run, ...cards.filter(card => card.tool === 'get_forecast').map(card => card.data)].find(run => run?.run_id === citation.run_id)
    if (forecast) return { title, forecast }
  }
  if (citation.kind === 'reference') {
    const fragments = cards.filter(card => card.tool === 'search_documents').flatMap(card => Array.isArray(card.data) ? card.data : [])
      .filter(fragment => fragment.reference_id === citation.reference_id && typeof fragment.text === 'string')
    if (fragments.length) return { title, fragments }
  }
  if (citation.kind === 'attribution_run' && evidence?.candidate_attribution?.run_id === citation.run_id) return { title, attribution: evidence.candidate_attribution }
  return null
}
