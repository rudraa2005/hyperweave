const $ = id => document.getElementById(id);
const fmt = (value, digits=3) => value == null ? 'not measured' : Number(value).toFixed(digits);
const pct = value => value == null ? 'not measured' : `${(100*value).toFixed(2)}%`;
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let overview, generated, busy=false;

function toast(message){const box=$('toast');box.textContent=message;box.style.display='block';setTimeout(()=>box.style.display='none',4500)}
async function get(url){const r=await fetch(url);const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data}
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data}
function card(label,value,note=''){return `<div class="stat"><small>${esc(label)}</small><b>${esc(value)}</b><small>${esc(note)}</small></div>`}
function activate(id){document.querySelectorAll('.view').forEach(el=>el.classList.toggle('active',el.id===id));document.querySelectorAll('nav button').forEach(el=>el.classList.toggle('active',el.dataset.view===id))}

function renderOverview(){
  const r=overview.report,p=r.partitions;
  $('hero-stats').innerHTML=card('A train / UNSW',p.a_train.rows.toLocaleString(),'group disjoint')+card('B evaluation / CSE',p.b.rows.toLocaleString(),'shift validation')+card('Sealed / CSE',p.sealed.rows.toLocaleString(),'opened once')+card('Operating point',pct(r.target_fpr),'A-val calibrated');
  $('limits-list').innerHTML=overview.limits.map(item=>`<li>${esc(item)}</li>`).join('');
  $('validator-summary').innerHTML=overview.validator_rejections?`<strong>${overview.validator_rejections} B-slice candidate validation rejections</strong><p class="muted">The coarse “worse” bucket means the B detection effect was below the predeclared validator cutoff. These are candidate checks, not C1–C4 claim verdicts.</p>`:'<strong>No candidate validation rejection occurred in this run.</strong><p class="muted">C1–C4 claims remain inconclusive.</p>';
  $('agent-events').innerHTML=overview.agent_events.map(e=>`<div class="event"><small>${esc(e.kind)} · event ${e.event_id}</small><pre>${esc(JSON.stringify(e.payload,null,2))}</pre></div>`).join('')||'<p>No recorded events available.</p>';
  renderScoreboard();
}

function renderScoreboard(){
  const r=overview.report, s=r.scoreboard;
  const rows=['b1','b2','b2b','b3','b4','b5','b5b'].map(name=>{
    const m=s[name], sealed=m.sealed?.original, recal=m.sealed?.recalibrated_diagnostic, ci=m.seed_bootstrap_detection_95;
    return `<tr><td><strong>${name.toUpperCase()}</strong></td><td>${m.completed}/${m.attempts}</td><td>${pct(m.mean_b_deployed_detection_rate??m.mean_b_detection_rate)}</td><td>${ci?`${pct(ci[0])}–${pct(ci[1])}`:'not measured'}</td><td>${pct(m.mean_b_deployed_fpr??m.mean_b_fpr)}</td><td>${fmt(m.mean_b_deployed_pr_auc??m.mean_b_pr_auc)}</td><td>${fmt(m.mean_macs,0)}</td><td>${m.mean_int8_model_bytes==null?'not measured':fmt(m.mean_int8_model_bytes,0)}</td><td>${m.mean_unpinned_kernel_ns_per_flow==null?'not measured':fmt(m.mean_unpinned_kernel_ns_per_flow,1)}</td><td>${fmt(m.b_hv_2d_debug_proxy,4)}</td><td>${fmt(m.research_hv_3d,4)}</td><td>${sealed?pct(sealed.detection_rate):'not measured'}</td><td>${m.sealed?.int8_original?pct(m.sealed.int8_original.detection_rate):'not measured'}</td><td>${sealed?pct(sealed.fpr):'not measured'}</td><td>${recal?pct(recal.detection_rate):'not measured'}</td></tr>`
  }).join('');
  $('scoreboard-table').innerHTML=`<table><thead><tr><th>Condition</th><th>Runs</th><th>B detection · deployed</th><th>95% seed interval</th><th>B FPR · deployed</th><th>B PR-AUC · deployed</th><th>Mean MACs</th><th>Int8 bytes</th><th>Kernel ns†</th><th>B HV · 2D proxy</th><th>B HV · 3D research</th><th>Sealed float detection</th><th>Sealed int8 detection</th><th>Sealed FPR</th><th>Sealed recal. detection*</th></tr></thead><tbody>${rows}</tbody></table><p class="muted">*Diagnostic only: recalibration uses sealed benign scores after final evaluation. It was not used to select or train a model. †${esc(r.latency_note||'Timing provenance unavailable')}.</p>`;
  const ae=r.offline_autoencoder;
  $('autoencoder').innerHTML=ae?.b?`<h3>Offline benign reconstruction diagnostic</h3><div class="cards">${card('B attack flags',pct(ae.b.b_metrics.detection_rate),'reconstruction error')}${card('B benign flags',pct(ae.b.b_metrics.fpr),'A-val benign threshold')}${card('Sealed attack flags',pct(ae.sealed?.original?.detection_rate),'once-only final evaluation')}${card('Sealed benign flags',pct(ae.sealed?.original?.fpr),'offline only')}</div><p class="muted">Backend: ${esc(ae.b.device)}. This autoencoder does not run in the int8 packet path; high benign flag rates on shifted traffic are a measured limitation.</p>`:'<h3>Offline benign reconstruction diagnostic</h3><p>not measured</p>';
  $('claims').innerHTML=`<h3>Research claim verdicts</h3><p>C1–C4: <strong>inconclusive</strong>. Original decision thresholds were unavailable before this study. The 2D B hypervolume is a debug proxy. ${esc(r.latency_note||'3D timing provenance unavailable')}.</p><p class="muted">Source: ${esc(r.registry_path)} · B3/B4/B5/B5b budget: ${Object.values(r.budget).join('/')} runs · empirical A-val FPR resolution: ${pct(r.a_val_empirical_fpr_resolution)}</p>`;
}

function renderGenerated(data){
  generated=data;
  $('model-cards').innerHTML=card('B detection · int8',pct(data.int8.detection_rate),'A-val threshold')+card('B false positives',pct(data.int8.fpr),'same threshold')+card('Model cost',data.macs.toLocaleString(),'MACs / flow')+card('Kernel latency',`${fmt(data.latency.median_ns_per_flow,1)} ns`,'unpinned debug')+card('Model size',`${data.int8_model_bytes} B`,'baked int8 artifact');
  $('neurons').innerHTML=data.gate_mask.map((open,i)=>`<span class="neuron ${open?'':'off'}" title="Neuron ${i+1}: ${open?'active':'pruned'}">${i+1}</span>`).join('');
  $('model-note').textContent=`Source ${data.source_run_id}; generated metric event ${data.registry_event_id}. Context ${data.context.join('/')}; optimizer steps during generation: ${data.optimizer_steps_during_generation}. Int8 detection gap: ${pct(data.int8_detection_gap)}. ${data.timing_note}.`;
  $('traffic-summary').innerHTML=card('B detection · int8',pct(data.int8.detection_rate),'selected generated candidate')+card('B FPR',pct(data.int8.fpr),'fixed A-val threshold')+card('B PR-AUC',fmt(data.int8.pr_auc),'attack ranking');
}

async function generate(){
  if(busy)return;
  busy=true;
  try{
    const cost=Number($('cost').value);$('cost-value').textContent=cost.toFixed(2);
    renderGenerated(await get(`/api/generate?cost=${cost}&load=${$('load').value}&alerts=${$('alerts').value}`));
    await traffic();
  }catch(e){toast(e.message)}finally{busy=false}
}
async function traffic(){
  const data=await get('/api/traffic?count=20');
  $('traffic-table').innerHTML=`<table><thead><tr><th>Flow</th><th>Known label</th><th>Model score</th><th>Decision</th></tr></thead><tbody>${data.sample.map(row=>`<tr><td>#${row.row+1}</td><td>${row.label?'Attack':'Benign'}</td><td>${fmt(row.score,4)}</td><td class="${row.alert?'alert':'normal'}">${row.alert?'Alert':'Pass'}</td></tr>`).join('')}</tbody></table>`;
}
async function evasion(){
  try{
    const data=await get(`/api/evasion?kind=${encodeURIComponent($('attack-kind').value)}&epsilon=${$('epsilon').value}`);
    $('evasion-result').innerHTML=card('Original detection',pct(data.original_detection_rate),`${data.attack_rows.toLocaleString()} attack flows`)+card('Under evasion',pct(data.evasion_detection_rate),`ε = ${data.epsilon}`)+card('Immutable fields','Preserved','endpoint / protocol');
  }catch(e){toast(e.message)}
}
async function autopilot(){
  try{
    const data=await get('/api/autopilot?window=128');
    const box=$('autopilot-result');box.hidden=false;
    box.innerHTML=`<h3>Context replay across ${data.windows} windows</h3><div class="cards">${card('Static detection',pct(data.static_detection_rate),`FPR ${pct(data.static_fpr)}`)}${card('Switched detection',pct(data.dynamic_detection_rate),`FPR ${pct(data.dynamic_fpr)}`)}${card('Static cost',fmt(data.static_macs_per_flow,0),'MACs / flow')}${card('Switched cost',fmt(data.dynamic_macs_per_flow,0),'MACs / flow')}</div><p class="muted">${data.context_changes} context changes. ${data.equal_detection_within_tolerance?`Cost change at matched detection: ${pct(data.cost_reduction_at_equal_detection)}.`:'Detection differed beyond the declared 2% tolerance, so no cost reduction claim is made.'} ${esc(data.note)}</p>`;
  }catch(e){toast(e.message)}
}
async function decide(approved){
  try{const data=await post('/api/decision',{approved});$('decision-result').textContent=JSON.stringify(data,null,2);toast(approved?'Simulation decision recorded':'Rejection recorded; execution prevented')}catch(e){toast(e.message)}
}

document.querySelectorAll('nav button').forEach(el=>el.addEventListener('click',()=>activate(el.dataset.view)));
$('limits-button').addEventListener('click',()=>$('limits').showModal());$('close-limits').addEventListener('click',()=>$('limits').close());
let timer;$('cost').addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(generate,220)});
$('load').addEventListener('change',generate);$('alerts').addEventListener('change',generate);
$('autopilot').addEventListener('click',autopilot);
$('run-evasion').addEventListener('click',evasion);$('approve').addEventListener('click',()=>decide(true));$('reject').addEventListener('click',()=>decide(false));
(async()=>{try{overview=await get('/api/overview');renderOverview();await generate()}catch(e){toast(e.message)}})();
