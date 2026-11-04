import { readFile, access } from 'node:fs/promises'
import { resolve } from 'node:path'

const root = resolve(process.env.FRONTEND_DIST_DIR || resolve(import.meta.dirname, '..', 'dist'))
const html = await readFile(resolve(root, 'index.html'), 'utf8')
if (/https?:\/\//.test(html)) throw new Error('构建入口包含外部资源地址')
const assets = [...html.matchAll(/(?:src|href)="([^"]+\.(?:js|css))"/g)].map(match => match[1])
if (!assets.some(path => path.endsWith('.js'))) throw new Error('缺少本地脚本资源')
for (const asset of assets) await access(resolve(root, asset.replace(/^\//, '')))
console.log(`构建检查通过：${assets.length} 个本地资源，无 CDN。`)
