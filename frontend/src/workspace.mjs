import { mineTime, observationWindow } from './utils.mjs'

export const modelKey = model => model ? `${model.model_id}:${model.version ?? model.model_version}` : ''

export function chooseModelKey(models, current = '') {
  if (models.some(model => modelKey(model) === current)) return current
  const defaults = models.filter(model => model.default_eligible !== false)
  const options = defaults.length ? defaults : models
  const latest = [...options].sort((a, b) => (Date.parse(b.created_at) || 0) - (Date.parse(a.created_at) || 0))[0]
  return modelKey(latest)
}

export function contextKey(session, pointId = '') {
  return JSON.stringify([session?.session_id, session?.source_id, session?.as_of, pointId])
}

export function createRequestGuard() {
  let current = 0
  return { begin: () => ++current, invalidate: () => { current++ }, isCurrent: value => value === current }
}

export function forecastMatches(run, session, pointId, selectedModel = '') {
  if (!run || !session || run.session_id !== session.session_id || run.source_id !== session.source_id || run.point_id !== pointId) return false
  if (selectedModel && modelKey(run) !== selectedModel) return false
  const end = Date.parse(session.as_of), asOf = Date.parse(run.as_of), cutoff = Date.parse(run.cutoff ?? run.as_of)
  return Number.isFinite(end) && Number.isFinite(asOf) && Number.isFinite(cutoff) && asOf <= end && cutoff <= end
}

export function forecastForEvent(current, evidence, session, pointId, selectedModel) {
  if (forecastMatches(evidence?.forecast_run, session, pointId)) return evidence.forecast_run
  return forecastMatches(current, session, pointId, selectedModel) ? current : null
}

export function latestVisibleReading(packet, asOf) {
  const cutoff = Date.parse(asOf)
  const rows = (packet?.rows || []).filter(row => {
    const eventTime = Date.parse(row.event_time), available = Date.parse(row.available_at || row.event_time), bucketEnd = Date.parse(row.bucket_end || row.event_time)
    return Number.isFinite(row.value) && Number.isFinite(eventTime) && eventTime <= cutoff && available <= cutoff && bucketEnd <= cutoff
      && !(row.quality_flags || []).some(flag => ['invalid', 'invalid_value'].includes(flag))
  })
  return rows.sort((a, b) => Date.parse(b.event_time) - Date.parse(a.event_time))[0] || null
}

export function displayWindow(asOf, event = null) {
  if (!event || !Number.isFinite(Date.parse(event.start_at))) return observationWindow(asOf)
  const start = Date.parse(event.start_at)
  const last = Math.max(start, Date.parse(event.end_at || event.event_time || event.start_at))
  return { start_at: new Date(start - 60 * 60000).toISOString(), end_at: new Date(Math.min(Date.parse(asOf), last + 10 * 60000)).toISOString() }
}

export function replayStart(coverage) {
  const start = Date.parse(coverage?.start_at), end = Date.parse(coverage?.end_at)
  if (!Number.isFinite(start) || !Number.isFinite(end)) return ''
  return mineTime(Math.min(start + 60000, end)).replace(' ', 'T')
}
