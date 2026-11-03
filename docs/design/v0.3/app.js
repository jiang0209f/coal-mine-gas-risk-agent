const root=document.querySelector('.app-shell');
const points={T0:{value:.86,base:.32,alarm:true,time:'08:39:55'},T1:{value:.56,base:.67,alarm:false,time:'08:39:52'},T2:{value:.42,base:.42,alarm:false,time:'08:39:54'},T5:{value:.38,base:.29,alarm:false,time:'08:39:51'}};
const nodes={T0:{x:561,y:269,dx:34,dy:10},T1:{x:317,y:312,dx:-124,dy:2},T2:{x:155,y:103,dx:-110,dy:-54},T5:{x:499,y:41,dx:32,dy:-12}};
const t0=[.32,.33,.35,.32,.34,.37,.36,.40,.39,.41,.45,.44,.46,.50,.49,.52,.56,.55,.58,.62,.64,.65,.69,.73,.75,.77,.81,.82,.84,.83,.86];
const series=Object.fromEntries(Object.entries(points).map(([id,p])=>[id,Array.from({length:31},(_,i)=>Number((id==='T0'?t0[i]:i===30?p.value:p.base+(p.value-p.base)*i/30+Math.sin(i*1.3)*.014).toFixed(3)))]));
const records=[{time:'04-07 08:32:48',point:'T0',type:'实测超限',level:'红色',value:'0.86',quality:'疑似标校',state:'待确认',active:true},{time:'04-07 07:12:00',point:'T0',type:'实测高位',level:'橙色',value:'0.69',quality:'—',state:'待复核',active:false},{time:'04-06 22:18:31',point:'T5',type:'实测超限',level:'红色',value:'0.81',quality:'已核实',state:'已复核',active:false},{time:'04-06 14:06:12',point:'T2',type:'实测高位',level:'橙色',value:'0.68',quality:'—',state:'已复核',active:false}];
const state={point:'T0',chart:'trend',tab:'analysis',filter:'all',confirmed:false,playing:true,offset:0,modal:null};
const modal=document.querySelector('.detail-modal');
let returnFocus=null;
const $=s=>document.querySelector(s);
const escapeHTML=text=>String(text).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clock=minutes=>`${String(Math.floor(minutes/60)%24).padStart(2,'0')}:${String(minutes%60).padStart(2,'0')}`;
function pointRows(){
 $('.point-list').innerHTML=Object.entries(points).map(([id,p])=>`<button class="point-row ${state.point===id?'active':''}" data-point="${id}" type="button" aria-pressed="${state.point===id}"><span class="point-id">${id}<small>${p.time}</small></span><strong class="point-value">${p.value.toFixed(2)}</strong><svg class="point-spark" viewBox="0 0 44 20" aria-label="${id}近期趋势"><path d="${series[id].slice(-10).map((v,i)=>(i?'L':'M')+(i*4.6).toFixed(1)+','+(18-(v-(id==='T0'?.6:.25))*24).toFixed(1)).join(' ')}" fill="none" stroke="${p.alarm?'#ee9ca5':'#6dbecc'}" stroke-width="1.4"/></svg><span class="point-state ${p.alarm?'alarm':''}">${p.alarm?'报警':'正常'}</span></button>`).join('');
 $('[data-selected-point]').textContent=state.point;$('[data-current]').textContent=points[state.point].value.toFixed(2);$('[data-face-point]').textContent=state.point;
 $('.concentration .summary-value').classList.toggle('danger-text',points[state.point].alarm);
 $('.concentration-status').textContent=points[state.point].alarm?'超限':'正常';$('.concentration-status').className='badge concentration-status '+(points[state.point].alarm?'danger':'success');
 $('[data-series-label]').textContent=state.point+' 实测';
 $('.map-labels').innerHTML=Object.entries(points).map(([id,p])=>`<button class="map-label" type="button" data-point="${id}" data-alarm="${p.alarm}" aria-pressed="${state.point===id}"><span>${id}</span><strong>${p.value.toFixed(2)}<small>%</small></strong></button>`).join('');
 placeMap();
}
function placeMap(){
 const scene=$('.face-scene'),w=scene.clientWidth,h=scene.clientHeight;if(!w||!h)return;
 const scale=Math.min(w/800,h/380),ox=(w-800*scale)/2,oy=(h-380*scale)/2;
 const project=(x,y)=>({x:ox+x*scale,y:oy+y*scale});
 const markup=[];
 for(const [id,n]of Object.entries(nodes)){
  const a=project(n.x,n.y),label=$(`.map-label[data-point="${id}"]`),lw=label.offsetWidth,lh=label.offsetHeight;
  const bx=Math.min(Math.max(12,a.x+n.dx),w-lw-12),by=Math.min(Math.max(12,a.y+n.dy),h-lh-12);
  label.style.left=bx+'px';label.style.top=by+'px';label.dataset.anchorX=a.x;label.dataset.anchorY=a.y;
  const edgeX=bx>a.x?bx:bx+lw,edgeY=by+lh/2;
  markup.push(`<path d="M${a.x},${a.y} L${(a.x+edgeX)/2},${a.y} L${edgeX},${edgeY}" fill="none" stroke="${points[id].alarm?'#a8667a':'#396f86'}" stroke-width="1"/><circle data-node="${id}" cx="${a.x}" cy="${a.y}" r="3.5" fill="${points[id].alarm?'#fa8b98':'#7bced6'}"/>`);
  if(state.point===id)markup.push(`<circle cx="${a.x}" cy="${a.y}" r="9" fill="none" stroke="#8ed7e2" stroke-width="1" opacity=".7"/>`);
 }
 const leaders=$('.face-leaders');leaders.setAttribute('viewBox',`0 0 ${w} ${h}`);leaders.innerHTML=markup.join('');
 for(const [selector,x,y,dx,dy]of [['.face-name',310,220,-35,-25],['.inlet-name',260,84,-18,-18],['.outlet-name',450,300,0,12],['.machine-name',475,140,-72,-19]]){
  const p=project(x,y),el=$(selector);el.style.left=p.x+dx+'px';el.style.top=p.y+dy+'px';
 }
}
function drawChart(){
 const plot=$('.trend-plot'),w=plot.clientWidth-20,h=plot.clientHeight-6;if(w<100||h<50)return;
 const pad={l:40,r:14,t:22,b:29},cw=w-pad.l-pad.r,ch=h-pad.t-pad.b,x=t=>pad.l+cw*t,y=v=>pad.t+ch*(1-v/1.06),endTime=8*60+40+state.offset;
 const axisColor='#8da4bf',gridColor='#263d52',tick=(tx,ty,text,anchor='middle')=>`<text x="${tx}" y="${ty}" fill="${axisColor}" font-size="12" text-anchor="${anchor}">${text}</text>`;
 let contents='';
 if(state.chart==='compare'){
  const rows=Object.keys(points),cellH=Math.min((ch-22)/4,40),cellW=cw/24,scale=['#173c56','#205b70','#3a8c99','#7abdb7','#cfb67b','#e5919b'];
  contents+=`<g data-heatmap>${rows.map((id,row)=>tick(pad.l-10,pad.t+row*cellH+cellH*.65,id,'end')+Array.from({length:24},(_,i)=>{const v=series[id][Math.round(i/23*30)],color=scale[v<.35?0:v<.45?1:v<.55?2:v<.65?3:v<.8?4:5];return`<rect x="${x(i/24)+.6}" y="${pad.t+row*cellH+.6}" width="${cellW-1.2}" height="${cellH-1.2}" fill="${color}"><title>${id} ${v.toFixed(2)}%</title></rect>`;}).join('')).join('')}</g>`;
  contents+=scale.map((c,i)=>`<rect x="${pad.l+i*14}" y="${pad.t+cellH*4+11}" width="13" height="5" fill="${c}"/>`).join('');
  contents+=tick(pad.l+91,pad.t+cellH*4+19,'低浓度　高浓度','start')+tick(pad.l,h-5,clock(endTime-60),'start')+tick(w-pad.r,h-5,clock(endTime),'end');
 }else{
  const cutoff=state.chart==='forecast'?.5:1;
  const values=series[state.point],path=values.map((v,i)=>(i?'L':'M')+x(i/30*cutoff).toFixed(1)+','+y(v).toFixed(1)).join(' ');
  const gridValues=h<145?[0,.4,.8]:[0,.4,.8,1];
  contents+=gridValues.map(v=>`<line x1="${pad.l}" x2="${w-pad.r}" y1="${y(v)}" y2="${y(v)}" stroke="${gridColor}" stroke-width=".7"/>`+tick(pad.l-8,y(v)+4,v.toFixed(1),'end')).join('');
  contents+=tick(7,14,'%CH₄','start');
  if(cutoff<1)contents+=`<rect x="${x(cutoff)}" y="${pad.t}" width="${cw*(1-cutoff)}" height="${ch}" fill="#091b2c99"/><line x1="${x(cutoff)}" x2="${x(cutoff)}" y1="${pad.t}" y2="${h-pad.b}" stroke="#536e86" stroke-dasharray="3 4"/>`+tick(x(.75),y(.42),'暂无预测结果');
  contents+=`<defs><linearGradient id="trend-fill" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#4bb6cf" stop-opacity=".18"/><stop offset="1" stop-color="#4bb6cf" stop-opacity="0"/></linearGradient></defs><path d="${path} L${x(cutoff)},${y(0)} L${pad.l},${y(0)}Z" fill="url(#trend-fill)"/><line x1="${pad.l}" x2="${w-pad.r}" y1="${y(.8)}" y2="${y(.8)}" stroke="#d78a95" stroke-width=".9" stroke-dasharray="4 4"/><path d="${path}" fill="none" stroke="#68c9dd" stroke-width="2"/><circle cx="${x(cutoff)}" cy="${y(points[state.point].value)}" r="3" fill="${points[state.point].alarm?'#fa8b98':'#68c9dd'}"/>`;
  const labels=cutoff<1?[[0,clock(endTime-60)],[.5,clock(endTime)],[1,clock(endTime+60)]]:(w<430?[0,.5,1]:[0,.25,.5,.75,1]).map(t=>[t,clock(endTime-60+t*60)]);
  contents+=labels.map(([p,label])=>tick(x(p),h-5,label,p===0?'start':p===1?'end':'middle')).join('');
 }
 plot.innerHTML=`<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${state.chart==='compare'?'测点浓度对比':state.point+'浓度趋势'}">${contents}</svg>`;
 root.querySelectorAll('[data-chart]').forEach(b=>b.classList.toggle('active',b.dataset.chart===state.chart));
 $('[data-chart-range]').textContent=state.chart==='forecast'?'预测 60 分钟':'近 60 分钟';
}
function renderInvestigation(){
 const body=$('.investigation-body');
 root.querySelectorAll('.investigation-tabs [data-tab]').forEach(b=>{const active=b.dataset.tab===state.tab;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));});
 if(state.tab==='analysis'){
  body.innerHTML=`<p class="analysis-lead">T0 浓度 <strong>0.86%</strong>，已超过 <strong>0.80%</strong> 报警值，当前保持红色预警。</p><h3>核查要点</h3><p class="body-text">核对传感器状态与标校记录，异常成因待现场核实。</p><div class="evidence-strip"><span>监测记录 31 条 · 规则 1 条</span><button type="button" data-tab="evidence">查看 ›</button></div><details class="task-details" ${innerHeight>920?'open':''}><summary>执行记录</summary><ul class="analysis-steps"><li><i class="step-check">✓</i>已读取监测窗口</li><li><i class="step-check">✓</i>已核对触发条件</li><li><i class="step-check">✓</i>已关联原始记录</li><li><i class="step-check waiting">4</i>处置条款待审核</li></ul></details><div class="analysis-message" hidden></div>`;
 }else if(state.tab==='evidence'){
  body.innerHTML=`<div class="evidence-grid"><div><small>当前测值</small>0.86%CH₄</div><div><small>报警值</small>0.80%CH₄</div></div><h3>监测片段</h3><table class="inside-table"><thead><tr><th>时间</th><th>浓度 / %</th><th>原始行号</th></tr></thead><tbody><tr><td>08:32:48</td><td>0.81</td><td>18236</td></tr><tr><td>08:36:10</td><td>0.84</td><td>18241</td></tr><tr><td>08:39:55</td><td>0.86</td><td>18248</td></tr></tbody></table><h3>触发规则</h3><p class="body-text">实测甲烷浓度达到0.80%，生成红色预警。</p><h3>质量标记</h3><span class="badge warning">疑似标校</span><p class="caption" style="margin-top:9px">事件修订 v2 · 预警保持可见</p>`;
 }else{
  body.innerHTML=`<div class="case-stages"><span class="active">确认</span><span>审核</span><span>执行</span><span>反馈</span><span>复核</span></div><h3>事件核查</h3><p class="body-text">核实监测记录、传感器状态和标校情况。</p><label class="caption">核查记录<textarea class="case-note" placeholder="填写核查结果"></textarea></label><button class="button primary" data-action="confirm" type="button">${state.confirmed?'事件已确认':'确认事件'}</button><div class="case-status" ${state.confirmed?'':'hidden'}>等待建议审核</div>`;
 }
}
function renderEvents(){
 $('.events-table tbody').innerHTML=records.filter(r=>state.filter==='all'||r.active).map(r=>`<tr><td>${r.time}</td><td>${r.point}</td><td>${r.type}</td><td><span class="badge ${r.level==='红色'?'danger':'warning'}">${r.level}</span></td><td>${r.value}</td><td>${r.quality}</td><td>${r.active&&state.confirmed?'已确认':r.state}</td><td><button type="button" data-action="${r.active?'focus-event':'review'}">查看</button></td></tr>`).join('');
 root.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b.dataset.filter===state.filter));
}
function openModal(type){
 returnFocus=document.activeElement;state.modal=type;
 const point=points[state.point];
 const entries={
  points:['监测点位',`<table class="inside-table"><thead><tr><th>测点</th><th>浓度 / %</th><th>状态</th><th>更新时间</th></tr></thead><tbody>${Object.entries(points).map(([id,p])=>`<tr><td><button class="text-button" data-modal-point="${id}" type="button">${id}</button></td><td>${p.value.toFixed(2)}</td><td>${p.alarm?'报警':'正常'}</td><td>${p.time}</td></tr>`).join('')}</tbody></table>`],
  point:['测点 '+state.point,`<div class="evidence-grid"><div><small>监测量</small>甲烷浓度</div><div><small>原始测值</small>${point.value.toFixed(2)}%CH₄</div><div><small>更新时间</small>${point.time}</div><div><small>状态</small>${point.alarm?'报警':'正常'}</div></div><p>工作面：31002<br>数据来源：历史甲烷监测<br>单位核对：待确认</p>`],
  quality:['导入数据质量','<div class="evidence-grid"><div><small>原始记录</small>770,930</div><div><small>有效记录</small>770,598</div><div><small>异常记录</small>332</div><div><small>有效占比</small>99.96%</div></div><h3>质量检查</h3><p>源文件校验已通过。异常值保留原值及行号，数值校验、时间校验和来源映射分别记录。</p>'],
  events:['预警记录',`<table class="inside-table"><thead><tr><th>时间</th><th>测点</th><th>等级</th><th>状态</th></tr></thead><tbody>${records.map(r=>`<tr><td>${r.time}</td><td>${r.point}</td><td>${r.level}</td><td>${r.active&&state.confirmed?'已确认':r.state}</td></tr>`).join('')}</tbody></table>`],
  advice:['处置建议','<h3>T0 甲烷浓度超限</h3><div class="evidence-grid"><div><small>事件编号</small>EV-0407-01</div><div><small>当前状态</small>等待资料审核</div></div><p>关联资料：31002工作面作业规程。</p><p>资料状态：待审核。</p><button class="button" data-open="knowledge" type="button" style="margin-top:16px">查看资料</button>'],
  knowledge:['知识资料','<h3>31002工作面作业规程</h3><div class="evidence-grid"><div><small>适用工作面</small>31002</div><div><small>资料状态</small>待审核</div></div><p>资料条款按原文、页码、版本和适用条件管理。</p><h3>处置经验</h3><p>按事件类型、工作面及复核状态查询。</p>'],
  models:['模型与评估','<div class="evidence-grid"><div><small>专业模型</small>未就绪</div><div><small>输入窗口</small>等待校验</div><div><small>模型产物</small>未登记</div><div><small>评估报告</small>暂无报告</div></div><p>事件查准率、召回率、每有效监测小时误报数和依据完整率由冻结评估记录生成。</p>'],
  system:['系统管理','<div class="evidence-grid"><div><small>当前工作面</small>31002</div><div><small>运行模式</small>历史回放</div><div><small>时间基准</small>UTC+8</div><div><small>当前角色</small>安全管理</div></div><p>数据源、监测点位、接口和权限设置。</p>'],
  review:['处置复核','<h3>T0 历史高位记录</h3><p>04-07 07:12—07:18，事件已结束。</p><label class="caption">复核意见<textarea class="case-note" placeholder="填写复核结论与依据"></textarea></label><button class="button primary" data-action="save-review" type="button">提交复核</button>']
 };
 const content=entries[type]||entries.point;$('[data-modal-title]').textContent=content[0];$('.modal-body').innerHTML=content[1];$('.modal-status').textContent='';if(!modal.open)modal.showModal();
}
function closeModal(){modal.close();returnFocus?.focus();}
document.addEventListener('click',event=>{
 const button=event.target.closest('button');if(!button)return;
 if(button.dataset.point){state.point=button.dataset.point;pointRows();drawChart();}
 else if(button.dataset.modalPoint){state.point=button.dataset.modalPoint;pointRows();drawChart();closeModal();}
 else if(button.dataset.chart){state.chart=button.dataset.chart;drawChart();}
 else if(button.dataset.tab){state.tab=button.dataset.tab;renderInvestigation();}
 else if(button.dataset.filter){state.filter=button.dataset.filter;renderEvents();}
 else if(button.dataset.open)openModal(button.dataset.open);
 else if(button.dataset.nav){const view=button.dataset.nav;root.querySelectorAll('[data-nav]').forEach(b=>b.classList.toggle('active',b===button));if(view==='events'||view==='cases'){state.tab=view==='cases'?'disposition':'evidence';renderInvestigation();}else if(view==='monitor'){state.chart='forecast';drawChart();}else if(view==='overview'){state.chart='trend';drawChart();}else openModal(view);}
 else if(button.dataset.action==='reset'){state.point='T0';pointRows();drawChart();}
 else if(button.dataset.action==='play'){state.playing=!state.playing;button.textContent=state.playing?'Ⅱ 暂停':'▷ 继续';}
 else if(button.dataset.action==='advance'){state.offset++;$('[data-clock]').textContent=clock(8*60+40+state.offset)+':00';drawChart();}
 else if(button.dataset.action==='focus-event'){state.tab='evidence';renderInvestigation();}
 else if(button.dataset.action==='review')openModal('review');
 else if(button.dataset.action==='advice')openModal('advice');
 else if(button.dataset.action==='save-review'){$('.modal-status').textContent='复核记录已保存';button.textContent='已提交';}
 else if(button.dataset.action==='confirm'){state.confirmed=true;renderInvestigation();renderEvents();$('[data-todo-summary]').textContent='待审核 1 · 待复核 1';}
 else if(button.hasAttribute('data-close')||button.classList.contains('dialog-close'))closeModal();
});
$('.question-form').addEventListener('submit',event=>{
 event.preventDefault();const input=$('.question-form input');if(!input.value.trim())return;
 state.tab='analysis';renderInvestigation();const message=$('.analysis-message');message.hidden=false;message.innerHTML='当前事件依据已整理：<br>T0 浓度0.86%，报警值0.80%。请结合原始记录核实标校状态。';input.value='';
});
modal.addEventListener('click',event=>{if(event.target===modal){const r=modal.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)closeModal();}});
modal.addEventListener('close',()=>returnFocus?.focus());
pointRows();renderInvestigation();renderEvents();drawChart();
new ResizeObserver(placeMap).observe($('.face-scene'));new ResizeObserver(drawChart).observe($('.trend-plot'));
window.addEventListener('resize',()=>{placeMap();drawChart();});
