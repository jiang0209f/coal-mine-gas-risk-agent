import { chromium } from '../../../frontend/node_modules/playwright/index.mjs';
import assert from 'node:assert/strict';
import { mkdir,writeFile } from 'node:fs/promises';
const output=process.env.DESIGN_QA_DIR||'E:/aqjs/design-review/v3';
await mkdir(output,{recursive:true});
const browser=await chromium.launch({executablePath:process.env.UI_TEST_BROWSER||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
const base=process.env.DESIGN_URL||'http://127.0.0.1:8012/';
const reports=[];
for(const [width,height] of [[1920,1080],[1440,900],[1366,768],[1024,900]]){
 await page.setViewportSize({width,height});await page.goto(base);await page.locator('.trend-plot svg').waitFor();
 await page.waitForTimeout(120);
 const result=await page.evaluate(()=>{
  const box=s=>{const r=document.querySelector(s).getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom};};
  const names=['.point-panel','.face-panel','.quality-panel','.trend-panel','.investigation-panel','.events-panel'];
  const panels=Object.fromEntries(names.map(s=>[s,box(s)]));
  const visible=[...document.querySelectorAll('button,th,td,h1,h2,h3,p,small,label,.body-text,.caption')].filter(e=>e.getClientRects().length&&e.textContent.trim());
  const tiny=visible.filter(e=>parseFloat(getComputedStyle(e).fontSize)<11.99).map(e=>({text:e.textContent.trim().slice(0,30),size:getComputedStyle(e).fontSize}));
  const banned=['设计预览','合成示例','集中监测 →','测点分组示意','全部数值','主屏持续展示'];
  const summary=box('.summary');
  const summaryBottom=Math.max(...[...document.querySelectorAll('.summary-item')].map(e=>e.getBoundingClientRect().bottom));
  const logRows=[...document.querySelectorAll('.events-table tbody tr')].map(e=>e.getBoundingClientRect().bottom);
  const overlaps=[];
  for(const name of document.querySelectorAll('.scene-name'))for(const label of document.querySelectorAll('.map-label')){
   const a=name.getBoundingClientRect(),b=label.getBoundingClientRect();
   if(a.left<b.right&&a.right>b.left&&a.top<b.bottom&&a.bottom>b.top)overlaps.push(name.textContent+' / '+label.textContent);
  }
  return {viewport:{width:innerWidth,height:innerHeight},scroll:{width:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight},panels,summary,summaryBottom,stage:box('.stage'),logWrap:box('.events-table-wrap'),logRows,overlaps,tiny,banned:banned.filter(s=>document.body.innerText.includes(s))};
 });
 assert.equal(result.scroll.width,width,JSON.stringify(result));assert.equal(result.scroll.height,height,JSON.stringify(result));
 assert.deepEqual(result.tiny,[]);assert.deepEqual(result.banned,[]);
 assert.deepEqual(result.overlaps,[],`${width}×${height} 地图名称与测点标签重叠`);
 assert(result.summaryBottom<=result.summary.bottom+.5,'摘要卡超出所在行');
 assert(result.stage.y-result.summaryBottom>=9,'摘要卡下方间距不足');
 const wholeRows=height<=820?2:3;
 assert(result.logRows[wholeRows-1]<=result.logWrap.bottom+.5,'预警记录末行截断');
 const p=result.panels;
 assert(Math.abs(p['.face-panel'].y-p['.investigation-panel'].y)<1);
 assert(Math.abs(p['.trend-panel'].bottom-p['.investigation-panel'].bottom)<1);
 if(width>=1280){assert(Math.abs(p['.point-panel'].y-p['.face-panel'].y)<1);assert(Math.abs(p['.quality-panel'].bottom-p['.trend-panel'].bottom)<1);}
 assert(p['.face-panel'].height>=205);assert(p['.trend-panel'].height>=170);
 assert(p['.events-panel'].bottom<=height-9);
 await page.screenshot({path:`${output}/dashboard-${width}x${height}.png`,fullPage:true});reports.push(result);
}
await page.setViewportSize({width:1440,height:900});await page.goto(base);
await page.locator('[data-point="T2"]').first().click();assert.equal(await page.locator('[data-selected-point]').innerText(),'T2');
await page.locator('[data-chart="compare"]').click();await page.locator('.trend-plot [data-heatmap]').waitFor();
await page.locator('[data-chart="forecast"]').click();await page.locator('.trend-plot').getByText('暂无预测结果').waitFor();
await page.locator('[data-chart="trend"]').click();
await page.locator('.investigation-tabs [data-tab="evidence"]').click();await page.locator('.investigation-body').getByText('触发规则').waitFor();
await page.locator('.investigation-tabs [data-tab="disposition"]').click();await page.locator('[data-action="confirm"]').click();
await page.locator('.investigation-body').getByText('等待建议审核').waitFor();
await page.locator('.investigation-tabs [data-tab="analysis"]').click();
await page.locator('[data-point="T0"]').first().click();
await page.screenshot({path:`${output}/interaction-1440x900.png`,fullPage:true});
for(const width of [768,390]){
 await page.setViewportSize({width,height:900});await page.goto(base);await page.locator('.trend-plot svg').waitFor();
 const size=await page.evaluate(()=>({viewport:innerWidth,width:document.documentElement.scrollWidth}));assert.equal(size.width,width);
 await page.screenshot({path:`${output}/dashboard-${width}.png`,fullPage:true});
}
assert.deepEqual(errors,[]);
const report={screens:reports,runtimeErrors:errors,interactionChecks:['point-selection','heatmap','forecast-state','evidence-tab','human-confirmation','responsive-width']};
await writeFile(`${output}/layout-report.json`,JSON.stringify(report,null,2));console.log(JSON.stringify({screens:reports.map(r=>({viewport:r.viewport,scroll:r.scroll})),runtimeErrors:errors}));
await browser.close();
