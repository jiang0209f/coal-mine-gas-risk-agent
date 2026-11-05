import assert from 'node:assert/strict'
import test from 'node:test'

test('chat deep links select only an explicitly requested session', async () => {
  const { chatDeepLink } = await import('../src/assistant.mjs')
  assert.deepEqual(chatDeepLink('?view=chat&session_id=authorized%2Freplay'), { view: 'chat', session_id: 'authorized/replay' })
  assert.equal(chatDeepLink(''), null)
  assert.equal(chatDeepLink('?view=chat'), null)
  assert.equal(chatDeepLink('?view=chat&session_id=%20'), null)
  assert.equal(chatDeepLink('?view=control&session_id=authorized-replay'), null)
})

test('chat routing distinguishes generic dialogue from data access and retains the authorized switch target', async () => {
  const { chatStatusView, assistantMessage, assistantChannel } = await import('../src/assistant.mjs')
  const status = { api_configured: true, mode: 'local_only', source_id: 'old-source', general_chat: true,
    available_sessions: [
      { session_id: 'other', source_id: 'authorized-fs', source_name: '风速历史' },
      { session_id: 'approved-replay', source_id: 'investigation-face-2022-t0', source_name: '已授权T0历史' },
    ] }
  const blocked = chatStatusView(status)
  assert.match(blocked.title, /通用对话可用/)
  assert.match(blocked.detail, /当前来源.*未.*API/)
  assert.equal(blocked.switchSession.session_id, 'approved-replay')
  assert.equal(chatStatusView({ ...status, api_configured: false, mode: 'not_configured' }).switchSession, null)
  assert.match(chatStatusView({ ...status, mode: 'summary' }).title, /当前来源已接入DeepSeek/)
  assert.equal(chatStatusView({ ...status, mode: 'summary' }).switchSession, null)
  const task = { task_id: 't', state: 'awaiting_input', method: 'structured_template', final_text: '请选择已授权回放',
    local_reason: 'cloud_source_not_authorized', error_code: 'cloud_source_not_authorized', chat_status: status, tool_steps: [] }
  const message = assistantMessage(task, {})
  assert.equal(message.text, task.final_text)
  assert.equal(message.local_reason, 'cloud_source_not_authorized')
  assert.equal(message.error_code, 'cloud_source_not_authorized')
  assert.deepEqual(message.chat_status, status)
  assert.equal(assistantChannel({ method: 'deepseek', model: 'deepseek-flash', chat_scope: 'general' }), 'DeepSeek通用对话')
})
