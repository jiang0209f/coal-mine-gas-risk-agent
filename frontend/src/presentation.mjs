export const presentationEnabled = import.meta.env?.VITE_PRESENTATION_MODE === 'anonymous'

const protectedFields = new Set(['password', 'password_hash', 'api_key', 'access_token', 'token', 'token_hash', 'authorization', 'raw_value', 'value'])
const escapePattern = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

function replacement(entries) {
  const values = new Map()
  for (const [source, target] of entries) {
    values.set(source, target)
    values.set(encodeURIComponent(source), encodeURIComponent(target))
  }
  const alternatives = [...values.keys()].sort((a, b) => b.length - a.length).map(value => {
    const left = /^[a-z0-9]/i.test(value) ? '(?<![a-z0-9])' : ''
    const right = /[a-z0-9]$/i.test(value) ? '(?![a-z0-9])' : ''
    return `${left}${escapePattern(value)}${right}`
  })
  const pattern = new RegExp(alternatives.join('|'), 'gi')
  // Keep replacement case-sensitive, including identifiers and URL escapes.
  return value => value.replace(pattern, matched => values.get(matched) ?? matched)
}

function walk(value, translate, field = '') {
  if (protectedFields.has(field.toLowerCase())) return value
  if (typeof value === 'string') return translate(value)
  if (Array.isArray(value)) return value.map(item => walk(item, translate, field))
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [translate(key), walk(item, translate, key)]))
  return value
}

export function createPresentationPolicy(aliases) {
  if (!Array.isArray(aliases) || !aliases.length || aliases.some(pair => !Array.isArray(pair) || pair.length !== 2 || pair.some(value => typeof value !== 'string' || !value.trim()) || pair[0] === pair[1])) throw new Error('匿名展示需要有效的工作面别名配置。')
  const originals = aliases.map(pair => pair[0]), labels = aliases.map(pair => pair[1])
  if (new Set(originals).size !== originals.length || new Set(labels).size !== labels.length || labels.some(label => originals.some(original => label.includes(original))) || originals.some(original => labels.some(label => original.includes(label)))) throw new Error('工作面别名重复或存在替换冲突。')
  const redactText = replacement(aliases)
  const restoreText = replacement(aliases.map(([original, alias]) => [alias, original]))
  return {
    redact: value => walk(value, redactText),
    restore: value => walk(value, restoreText),
    restorePath: restoreText,
    restoreForm(body) {
      const restored = new FormData()
      for (const [key, value] of body.entries()) restored.append(key, typeof value === 'string' ? restoreText(value) : value)
      return restored
    },
  }
}

export function configuredPresentationAliases() {
  try { return JSON.parse(import.meta.env?.VITE_PRESENTATION_ALIASES || '[]') } catch { throw new Error('工作面别名配置格式错误。') }
}

export async function anonymousReportBlob(data) {
  const sourceDigests = new Set(['report_sha256', 'manifest_sha256', 'records_sha256', 'export_records_sha256', 'export_manifest_sha256'])
  function displayRecords(value) {
    if (Array.isArray(value)) return value.map(displayRecords)
    if (!value || typeof value !== 'object') return value
    return Object.fromEntries(Object.entries(value)
      .filter(([key]) => key !== 'projection_sha256' && key !== 'export_sha256')
      .map(([key, item]) => [sourceDigests.has(key) ? `source_${key}` : key, displayRecords(item)]))
  }
  const records = displayRecords(data)
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(records)))
  const displaySha256 = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('')
  const exported = { format: 'anonymous-display-v1', source_report_sha256: data.report_sha256 || data.report?.report_sha256 || null, records, display_sha256: displaySha256 }
  return new Blob([JSON.stringify(exported, null, 2)], { type: 'application/json;charset=utf-8' })
}
