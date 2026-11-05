<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { createModelConnectionClient } from '../model-connection.mjs'

const props = defineProps({ api: { type: Function, required: true } })
const client = createModelConnectionClient(props.api)
const configuration = ref(null), baseUrl = ref('https://api.deepseek.com'), model = ref('deepseek-flash'), apiKey = ref('')
const working = ref(''), error = ref(''), notice = ref(''), result = ref(null)
let mounted = true
const changed = computed(() => Boolean(apiKey.value.trim()) || baseUrl.value.trim() !== configuration.value?.base_url || model.value.trim() !== configuration.value?.model)
const canTest = computed(() => !working.value && configuration.value?.configured && !changed.value)
function apply(value) {
  configuration.value = value
  baseUrl.value = value.base_url; model.value = value.model
}
async function load() {
  working.value = 'load'; error.value = ''; notice.value = ''; result.value = null
  try {
    const value = await client.load()
    if (mounted) apply(value)
  } catch (caught) { if (mounted) error.value = caught.message }
  finally { if (mounted) working.value = '' }
}
async function save() {
  if (working.value) return
  const key = apiKey.value
  apiKey.value = ''; working.value = 'save'; error.value = ''; notice.value = ''; result.value = null
  try {
    const value = await client.save({ base_url: baseUrl.value, model: model.value, api_key: key })
    if (mounted) { apply(value); notice.value = '模型连接配置已保存。' }
  } catch (caught) {
    if (mounted) error.value = key.trim() ? caught.message.split(key.trim()).join('[已隐藏]') : caught.message
  } finally { if (mounted) working.value = '' }
}
async function test() {
  if (!canTest.value) return
  working.value = 'test'; error.value = ''; notice.value = ''; result.value = null
  try {
    const value = await client.test()
    if (mounted) result.value = value
  } catch (caught) { if (mounted) error.value = caught.message }
  finally { if (mounted) working.value = '' }
}
onMounted(load)
onBeforeUnmount(() => { mounted = false; apiKey.value = '' })
</script>

<template>
  <section class="model-connection panel" aria-label="模型连接设置">
    <header class="connection-heading"><div><h2>模型连接</h2><p>配置大模型服务与连接测试。</p></div><span :class="['connection-state', { configured: configuration?.configured }]">{{ working === 'load' ? '读取中' : configuration?.configured ? '密钥已配置' : '未配置密钥' }}</span></header>
    <form class="connection-form" @submit.prevent="save">
      <label>服务方式<select value="deepseek" :disabled="!!working" aria-label="模型服务方式"><option value="deepseek">DeepSeek API</option><option value="local" disabled>本地兼容接口（预留）</option></select></label>
      <label>接口地址<input v-model="baseUrl" type="url" required :disabled="!!working" autocomplete="off" spellcheck="false" aria-label="DeepSeek接口地址"></label>
      <label>模型名称<input v-model="model" required :disabled="!!working" autocomplete="off" spellcheck="false" aria-label="DeepSeek模型名称"></label>
      <label>API Key<input v-model="apiKey" type="password" :disabled="!!working" autocomplete="off" spellcheck="false" :placeholder="configuration?.configured ? '留空保留当前密钥' : '请输入DeepSeek API Key'" aria-label="DeepSeek API Key"><small>留空保留当前密钥，保存后输入框会清空。</small></label>
      <div v-if="configuration" class="connection-metadata"><span>当前配置：{{ configuration.source === 'saved' ? '本机保存' : '环境配置' }}</span><span>{{ configuration.model }}</span></div>
      <div class="connection-actions"><button class="primary" type="submit" :disabled="!!working || !baseUrl.trim() || !model.trim()">{{ working === 'save' ? '保存中…' : '保存配置' }}</button><button class="secondary" type="button" :disabled="!canTest" @click="test">{{ working === 'test' ? '连接测试中…' : '测试连接' }}</button><button v-if="error && !configuration" class="ghost" type="button" :disabled="!!working" @click="load">重新读取</button></div>
      <p class="connection-note">{{ changed && configuration ? '请先保存修改，再测试连接。' : '连接测试使用已保存配置和固定测试消息。' }}</p>
    </form>
    <p v-if="error" class="connection-message error-message" role="alert">{{ error }}</p><p v-if="notice" class="connection-message success-message" role="status">{{ notice }}</p>
    <section v-if="result" :class="['connection-result', { success: result.ok === true }]" aria-label="模型连接测试结果" role="status"><strong>{{ result.ok === true ? '连接成功' : '连接未成功' }}</strong><p>{{ result.message }}</p><div v-if="result.model || Number.isFinite(result.latency_ms)"><span v-if="result.model">{{ result.model }}</span><span v-if="Number.isFinite(result.latency_ms)">{{ result.latency_ms }} ms</span></div></section>
  </section>
</template>

<style scoped>
.model-connection{max-width:880px;width:100%;padding:24px;color:#c5d9e8}.connection-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;margin-bottom:24px}.connection-heading h2{font-size:18px;font-weight:500}.connection-heading p{font-size:13px;color:#91abc2;margin-top:7px}.connection-state{padding:3px 9px;border:1px solid #66543e;border-radius:4px;font-size:12px;color:#e6b978;white-space:nowrap}.connection-state.configured{color:#8cd4c1;border-color:#34685a}.connection-form{display:grid;gap:18px;max-width:620px}.connection-form>label{font-size:13px;color:#a4bed0}.connection-form input,.connection-form select{height:40px;background:#0b2134;border:1px solid #34546c;color:#d1e4f1;padding:8px 11px;border-radius:5px;font-size:14px;margin-top:7px}.connection-form input::placeholder{color:#7795ac}.connection-form small{display:block;font-size:12px;color:#8ca8bf;margin-top:6px}.connection-form option:disabled{color:#68859b}.connection-metadata{display:flex;gap:18px;flex-wrap:wrap;color:#8faec4;font-size:12px}.connection-actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.connection-actions button{min-height:36px;font-size:13px}.connection-note{font-size:12px;color:#8fa9bf;margin-top:-7px}.connection-message{font-size:13px;line-height:1.8;margin-top:20px;overflow-wrap:anywhere}.error-message{color:#edb198}.success-message{color:#8cd4c1}.connection-result{margin-top:22px;max-width:620px;padding:14px 16px;border:1px solid #66543e;background:#302a242e;border-radius:5px;color:#e6b978;font-size:13px}.connection-result.success{border-color:#316458;background:#163e3733;color:#9cdac8}.connection-result strong{font-size:14px;font-weight:500}.connection-result p{margin-top:6px;line-height:1.8;overflow-wrap:anywhere}.connection-result>div{display:flex;gap:18px;color:#91abc2;margin-top:8px;font-size:12px}@media(max-width:760px){.model-connection{padding:18px}.connection-heading{gap:10px}.connection-heading h2{font-size:17px}.connection-form{gap:15px}.connection-actions{gap:8px}}
</style>
