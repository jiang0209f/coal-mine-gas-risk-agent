import assert from 'node:assert/strict'
import test from 'node:test'

const assistant = () => import('../src/assistant.mjs')

test('assistant channel stays neutral until a provider has actually returned a model', async () => {
  const { assistantChannel } = await assistant()
  assert.equal(assistantChannel(), '业务查询')
  assert.equal(assistantChannel({ method: 'deepseek', model: null, state: 'queued' }), '业务查询')
  assert.equal(assistantChannel({ method: 'deepseek', model: null, state: 'failed' }), '业务查询')
  assert.equal(assistantChannel({ model: 'deepseek-flash' }), '业务查询')
})

test('local tool answers retain their real channel even if a stale model name is present', async () => {
  const { assistantChannel } = await assistant()
  assert.equal(assistantChannel({ method: 'structured_template', model: 'deepseek-flash' }), '本机查询')
  assert.equal(assistantChannel({ brief_method: 'structured_template' }), '本机查询')
  assert.equal(assistantChannel({ method: 'deepseek', model: 'deepseek-flash', state: 'completed' }), 'DeepSeek')
})

test('a source-policy block remains a local result even when a cloud task was initially scheduled', async () => {
  const { assistantChannel } = await assistant()
  assert.equal(assistantChannel({ method: 'deepseek', model: null, state: 'failed', error_code: 'cloud_source_not_authorized' }), '本机查询')
})

test('tool steps present business names and preserve failed or cancelled states', async () => {
  const { toolName, toolState, toolArguments } = await assistant()
  assert.equal(toolName('query_series'), '查询监测序列')
  assert.equal(toolName('search_documents'), '检索批准资料')
  assert.equal(toolName('unknown_internal_tool'), '业务操作')
  assert.equal(toolState('failed'), '失败')
  assert.equal(toolState('cancelled'), '已取消')
  assert.equal(toolState('success'), '已完成')
  assert.equal(toolArguments({ point_id: '测点A', query: '调校记录', raw_private_path: 'PRIVATE-PATH' }), '测点：测点A · 关键词：调校记录')
  assert.equal(toolArguments({ query: { invalid: true }, hidden: 'PRIVATE' }), '')
})

test('presentation reference download uses displayed excerpts and never requests the original file', async () => {
  const { prepareReferenceDownload } = await assistant()
  let requested = false
  const result = await prepareReferenceDownload({ presentation: true,
    document: { title: '采煤工作面作业规程', version: '现行版', source: '本机资料库',
      fragments: [{ text: '核对测点记录。', locator: { page: 3 } }, { text: '填写调校记录。', locator: { page: 9 } }] },
    file: { file_id: 'PRIVATE-ORIGINAL-FILE', filename: 'PRIVATE-ORIGINAL.pdf' },
    api: () => { requested = true; throw new Error('original file must not be requested') } })
  assert.equal(requested, false)
  assert.equal(result.filename, '采煤工作面作业规程-摘录.txt')
  const text = await result.blob.text()
  assert.match(text, /采煤工作面作业规程/)
  assert.match(text, /第3页\n核对测点记录。/)
  assert.match(text, /第9页\n填写调校记录。/)
  assert.doesNotMatch(text, /PRIVATE-ORIGINAL/)
})

test('normal reference download retains the authenticated original-file path and content', async () => {
  const { prepareReferenceDownload } = await assistant()
  const original = new Blob(['original bytes'], { type: 'application/pdf' })
  const calls = []
  const result = await prepareReferenceDownload({ presentation: false,
    file: { file_id: 'file with spaces', filename: '作业规程.pdf' },
    api: async (...args) => { calls.push(args); return original } })
  assert.deepEqual(calls, [['/references/files/file%20with%20spaces/content', { responseType: 'blob' }]])
  assert.equal(result.blob, original)
  assert.equal(result.filename, '作业规程.pdf')
})

test('presentation export reports a missing excerpt instead of falling back to the original file', async () => {
  const { prepareReferenceDownload } = await assistant()
  let requested = false
  await assert.rejects(() => prepareReferenceDownload({ presentation: true,
    document: { title: '资料', fragments: [] }, file: { file_id: 'original' },
    api: () => { requested = true } }), /暂无可导出的资料摘录/)
  assert.equal(requested, false)
})
