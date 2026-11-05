const metadata = value => ({
  provider: 'deepseek', base_url: value?.base_url || 'https://api.deepseek.com',
  model: value?.model || 'deepseek-flash', configured: value?.configured === true,
  source: value?.source === 'saved' ? 'saved' : 'environment', local_available: false,
})

export function createModelConnectionClient(api) {
  return {
    load: async () => metadata(await api('/model-connection')),
    save: async input => {
      const body = { provider: 'deepseek', base_url: input.base_url.trim(), model: input.model.trim() }
      if (!body.base_url || !body.model) throw new Error('请填写接口地址和模型名称')
      if (input.api_key?.trim()) body.api_key = input.api_key.trim()
      return metadata(await api('/model-connection', { method: 'PUT', body }))
    },
    test: () => api('/model-connection/test', { method: 'POST' }),
  }
}
