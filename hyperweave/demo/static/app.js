const $ = id => document.getElementById(id);
const fmt = (value, digits=3) => value == null ? 'not measured' : Number(value).toFixed(digits);
const pct = value => value == null ? 'not measured' : `${(100*value).toFixed(2)}%`;
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let overview, generated, busy=false;
let liveAfter=0, ratePoints=[], livePollTimer, captureErrorShown=false;
let domainComparison=null;

function toast(message){const box=$('toast');box.textContent=message;box.style.display='block';setTimeout(()=>box.style.display='none',4500)}
async function get(url){const r=await fetch(url);const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data}
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data}
function card(label,value,note=''){return `<div class="stat"><small>${esc(label)}</small><b>${esc(value)}</b><small>${esc(note)}</small></div>`}
function activate(id){document.querySelectorAll('.view').forEach(el=>el.classList.toggle('active',el.id===id));document.querySelectorAll('nav button').forEach(el=>el.classList.toggle('active',el.dataset.view===id));if(id==='posture')requestAnimationFrame(drawDomainChart)}

function renderOverview(){
  const r=overview.report,p=r.partitions;
  $('hero-stats').innerHTML=card('A train / UNSW',p.a_train.rows.toLocaleString(),'group disjoint')+card('B evaluation / CSE',p.b.rows.toLocaleString(),'shift validation')+card('Sealed / CSE',p.sealed.rows.toLocaleString(),'opened once')+card('Operating point',pct(r.target_fpr),'A-val calibrated');
  $('limits-list').innerHTML=overview.limits.map(item=>`<li>${esc(item)}</li>`).join('');
  renderNotebook();
  renderScoreboard();
}

const roles=[['Coordinator','Approval and budget gate'],['Neuron Architect','Architecture and parameter profile'],['Training Strategist','Training configuration and diagnostics'],['Optimization Scientist','Pareto and hypervolume analysis'],['Experimental Scientist','Experiment planning and proposal'],['Adversarial Validator','Coarse robustness verdict']];
function renderNotebook(){
  const events=overview.agent_events,calls=events.filter(event=>event.kind==='agent_call');
  const proposal=events.find(event=>event.kind==='proposal')?.payload.proposal;
  const decision=events.find(event=>event.kind==='human_decision')?.payload;
  const outcome=events.find(event=>event.kind==='cycle_outcome')?.payload;
  const verdict=events.find(event=>event.kind==='validator_verdict')?.payload;
  const unlinked=overview.unlinked_agent_calls||[];
  const traces=[...calls.map(event=>({...event,linked:true})),...unlinked.map(event=>({...event,kind:'agent_call',payload:event,linked:false}))];
  traces.sort((left,right)=>Number(left.event_id)-Number(right.event_id));
  $('validator-summary').innerHTML=card('Recorded proposal',proposal?.action||'not recorded',proposal?.hypothesis||'No proposal detail')+
    card('Human gate',decision?(decision.approved?'approved':'rejected'):'not recorded',decision?.actor||'No decision record')+
    card('Execution',outcome?.status||'not recorded',outcome?.executed?'action executed':'no action execution')+
    card('Validator',verdict?`${verdict.verdict} · ${verdict.bucket}`:'not recorded','coarse B-slice result')+
    card('Linked role calls',calls.length,`${unlinked.length} registry calls lack proposal linkage`);
  const stages=[['agent_call','Role call'],['proposal','Proposal'],['human_decision','Human decision'],['action_started','Execution started'],['action_status','Execution status'],['validator_verdict','Validation'],['cycle_outcome','Outcome']];
  const ordered=events.map(event=>({event,label:stages.find(([kind])=>kind===event.kind)?.[1]||event.kind.replaceAll('_',' ')}));
  $('agent-events').innerHTML=ordered.length?ordered.map(({event,label},index)=>{
    const payload=event.payload;let title=label,description='';
    if(event.kind==='agent_call'){title=`${payload.role} reviewed the proposal`;description=payload.prompt||'Recorded role review'}
    if(event.kind==='proposal'){title=payload.proposal.action.replaceAll('_',' ');description=payload.proposal.hypothesis||payload.proposal.rationale}
    if(event.kind==='human_decision'){title=payload.approved?'Approval recorded':'Proposal rejected';description=`Decision by ${payload.actor}`}
    if(event.kind==='action_started'){title='Approved action started';description=`Proposal ${payload.proposal_id}`}
    if(event.kind==='action_status'){title=`Action ${payload.status}`;description=`Proposal ${payload.proposal_id}`}
    if(event.kind==='validator_verdict'){title=`${payload.verdict} · ${payload.bucket}`;description=`Coarse validator result for ${payload.run_id}`}
    if(event.kind==='cycle_outcome'){title=payload.status.replaceAll('_',' ');description=payload.executed?'Recorded action execution completed':payload.approved?'Approved; execution recorded':'No action executed'}
    return `<article class="workflow-event"><span class="workflow-step">${String(index+1).padStart(2,'0')}</span><div class="workflow-copy"><div class="workflow-title"><strong>${esc(title)}</strong><small>event ${event.event_id}</small></div><p>${esc(description||'No additional detail recorded')}</p><details><summary>Event details</summary><pre>${esc(JSON.stringify(payload,null,2))}</pre></details></div></article>`
  }).join(''):'<p class="empty-state">No proposal-linked workflow events were found in this registry.</p>';
  $('unlinked-calls').hidden=!unlinked.length;
  $('unlinked-calls').querySelector('summary').textContent=`Unlinked historical agent calls · ${unlinked.length}`;
  $('unlinked-agent-events').innerHTML=unlinked.map(event=>`<article class="unlinked-call"><strong>${esc(event.role||'Unknown role')}</strong><span>event ${event.event_id} · ${esc(event.model_version||'model not recorded')}</span><p>${esc(event.prompt||'No prompt recorded')}</p><details><summary>Unlinked call data</summary><pre>${esc(JSON.stringify({temperature:event.temperature,tool_results:event.tool_results,response:event.response},null,2))}</pre></details></article>`).join('');
  $('agent-team').innerHTML=roles.map(([name,job],index)=>{const matching=traces.filter(event=>event.payload.role===name);const latest=matching.at(-1);return `<button class="agent-card ${matching.length?'has-trace':'no-trace'}" data-role="${esc(name)}"><span class="role-index">0${index+1}</span><span class="agent-activity-mark"><i></i></span><span class="agent-card-copy"><strong>${esc(name)}</strong><small>${esc(job)}</small></span><span class="role-count ${matching.length?'has-calls':''}">${matching.length?`${matching.length} recorded call${matching.length===1?'':'s'} · ${latest?.linked?'proposal linked':'historical trace'}`:'No recorded call'}</span></button>`}).join('');
  document.querySelectorAll('.agent-card').forEach(button=>button.addEventListener('click',()=>{
    const entries=traces.filter(e=>e.payload.role===button.dataset.role);
    const job=roles.find(([name])=>name===button.dataset.role)?.[1]||'';
    $('agent-detail').innerHTML=`<p class="eyebrow">ROLE INSPECTOR</p><h3>${esc(button.dataset.role)}</h3><p class="muted">${esc(job)}</p>`+(entries.length?entries.map(event=>`<article class="role-call"><span>${event.linked?'Proposal-linked':'Unlinked historical'} registry call · event ${event.event_id}</span><h4>Task</h4><p>${esc(event.payload.prompt||'Role review')}</p><h4>Recorded response</h4><pre>${esc(JSON.stringify(event.payload.response,null,2))}</pre><details><summary>Tool evidence</summary><pre>${esc(JSON.stringify(event.payload.tool_results,null,2))}</pre></details><small>${esc(event.payload.model_version||'scripted replay')} · temperature ${fmt(event.payload.temperature,2)}</small></article>`).join(''):'<div class="role-empty">No registry call is recorded for this role. The notebook leaves it idle instead of implying it ran.</div>');
    document.querySelectorAll('.agent-card').forEach(el=>el.classList.toggle('selected',el===button));
  }));
  $('agent-detail').innerHTML='<p class="eyebrow">ROLE INSPECTOR</p><h3>Select a role</h3><p class="muted">Open a role to inspect its recorded task, tool evidence, and response.</p>';
  $('replay-agents').onclick=()=>replayAgentTraces(traces);
}

let agentReplayToken=0;
async function replayAgentTraces(traces){
  const button=$('replay-agents'),feed=$('agent-playback'),token=++agentReplayToken;
  const sequence=roles.map(([name])=>({name,event:traces.filter(item=>item.payload.role===name).at(-1)}));
  document.querySelectorAll('.agent-card').forEach(card=>card.classList.remove('working','completed','selected'));
  button.disabled=true;
  feed.innerHTML=`<div class="replay-header"><strong>Recorded-work playback</strong><span>${sequence.filter(item=>item.event).length} roles with traces · ${sequence.filter(item=>!item.event).length} without a recorded call</span></div><div class="replay-track"><i></i></div><p class="replay-current">Reading the saved role sequence…</p>`;
  for(let index=0;index<sequence.length;index++){
    if(token!==agentReplayToken)return;
    const {name,event}=sequence[index],card=[...document.querySelectorAll('.agent-card')].find(el=>el.dataset.role===name),progress=feed.querySelector('.replay-track i'),current=feed.querySelector('.replay-current');
    if(progress)progress.style.width=`${Math.round((index/sequence.length)*100)}%`;
    if(!event){current.textContent=`${name}: no call in the registry; left idle.`;await new Promise(resolve=>setTimeout(resolve,430));continue}
    card?.classList.add('working');
    current.innerHTML=`<b>${esc(name)}</b> · inspecting event ${esc(event.event_id)} · ${event.linked?'proposal-linked':'historical trace'}`;
    await new Promise(resolve=>setTimeout(resolve,650));
    if(token!==agentReplayToken)return;
    const tools=event.payload.tool_results||{};
    const evidence=Object.keys(tools).length?Object.keys(tools).map(key=>`<span>${esc(key.replaceAll('_',' '))}</span>`).join(''):'No tool inputs recorded';
    feed.insertAdjacentHTML('beforeend',`<article class="replay-result"><div><b>${esc(name)}</b><small>event ${esc(event.event_id)} · ${esc(event.payload.model_version||'recorded client')}</small></div><p>${esc(event.payload.prompt||'Role task')}</p><div class="replay-tools">${evidence}</div><details><summary>Inspect recorded response</summary><pre>${esc(JSON.stringify(event.payload.response,null,2))}</pre></details></article>`);
    card?.classList.remove('working');card?.classList.add('completed');
    await new Promise(resolve=>setTimeout(resolve,400));
  }
  if(token===agentReplayToken){const progress=feed.querySelector('.replay-track i');if(progress)progress.style.width='100%';feed.querySelector('.replay-current').textContent='Playback complete · displayed tasks and outputs came from saved registry events.';button.disabled=false}
}

const mapDescriptions={
  flows:'Live mode reads packets from the selected interface and aggregates headers into approximate bidirectional flows. Recorded benchmark mode reads held-out NF-v2 flow rows. Both use the study normalizer; live packets have no ground-truth labels.',
  preference:'λ controls the tradeoff among task loss, shift loss, and expected inference cost. Discrete context bins encode link load and alert rate.',
  hypernet:'The B3 hypernetwork conditions learned neuron embeddings on λ and context, then emits functional-neuron parameters and hard-concrete gates.',
  bake:'The selected parameters are generated once, closed neurons are removed, and the model is quantized to int8. The slider runs generation without optimizer updates.',
  kernel:'Recorded evaluation sends the exported model through the C kernel and verifies bit-exact logits. Live capture scores with the same generated int8 weights in Python for low-latency packet updates. The live feature adapter is approximate; offline artifact hashes enter the local registry.'
};
function selectMapNode(node){document.querySelectorAll('.map-node').forEach(el=>el.classList.toggle('selected',el===node));$('map-detail').textContent=mapDescriptions[node.dataset.node]}

async function loadCaptureInterfaces(){
  try{
    const data=await get('/api/live/interfaces');
    const select=$('capture-interface');
    select.innerHTML=data.interfaces.map(item=>`<option value="${esc(item.name)}">${esc(item.name)} · ${esc(item.status)}</option>`).join('')||'<option value="">No interfaces found</option>';
    if(!data.available)$('capture-note').textContent=data.error;
  }catch(error){$('capture-note').textContent=`Could not list capture interfaces: ${error.message}`}
}
function drawRateChart(){
  const canvas=$('rate-chart');if(!canvas)return;
  const rect=canvas.getBoundingClientRect(), scale=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.floor(rect.width*scale));canvas.height=Math.floor(170*scale);
  const context=canvas.getContext('2d');context.scale(scale,scale);
  const width=rect.width,height=170,pad=22;
  context.clearRect(0,0,width,height);context.strokeStyle='#e7e7e7';context.lineWidth=1;
  for(let row=0;row<4;row++){const y=pad+row*(height-pad*2)/3;context.beginPath();context.moveTo(pad,y);context.lineTo(width-pad,y);context.stroke()}
  if(ratePoints.length<2)return;
  for(const [key,color] of [['pps','#111'],['mbps','#888']]){
    const max=Math.max(1,...ratePoints.map(point=>point[key]));context.strokeStyle=color;context.lineWidth=2;context.beginPath();
    ratePoints.forEach((point,index)=>{const x=pad+index*(width-pad*2)/(ratePoints.length-1),y=height-pad-(point[key]/max)*(height-pad*2);if(index===0)context.moveTo(x,y);else context.lineTo(x,y)});context.stroke();
  }
}
function renderLive(data){
  $('capture-state').textContent=data.error?`Capture failed · ${data.error}`:data.running?(data.ready?`Capturing · ${data.interface}`:`Starting · ${data.interface}`):'Capture stopped';
  $('capture-start').disabled=data.running;$('capture-stop').disabled=!data.running;
  $('live-cards').innerHTML=card('Packets',Number(data.packets).toLocaleString(),'captured at interface')+card('Flows',Number(data.flows).toLocaleString(),'bidirectional 5-tuple aggregates')+card('Packet rate',`${fmt(data.packets_per_second,1)} pkt/s`,'rolling capture rate')+card('Network rate',`${fmt(data.bits_per_second/1e6,3)} Mbit/s`,'captured bytes · link overhead excluded')+card('Model alerts',Number(data.alerts).toLocaleString(),'no packet ground truth');
  $('capture-note').textContent=data.feature_note;
  ratePoints.push({pps:data.packets_per_second,mbps:data.bits_per_second/1e6});if(ratePoints.length>60)ratePoints.shift();
  drawRateChart();
  $('rate-legend').textContent=`Latest ${fmt(data.packets_per_second,1)} packets/s · ${fmt(data.bits_per_second/1e6,3)} Mbit/s · values rescale independently in the chart.`;
  if(data.events.length){
    liveAfter=data.events[data.events.length-1].id;
    const rows=data.events.slice(-18).reverse();
    $('live-flow-table').innerHTML=`<table><thead><tr><th>Time</th><th>Flow</th><th>Protocol</th><th>Packets</th><th>Bytes</th><th>Model score</th><th>Model output</th></tr></thead><tbody>${rows.map(event=>`<tr><td>${new Date(event.timestamp*1000).toLocaleTimeString()}</td><td>${esc(event.src)}:${event.src_port} → ${esc(event.dst)}:${event.dst_port}</td><td>${event.protocol===6?'TCP':event.protocol===17?'UDP':`IP ${event.protocol}`}</td><td>${event.flow_packets}</td><td>${Number(event.flow_bytes).toLocaleString()}</td><td>${fmt(event.score,4)}</td><td class="${event.alert?'alert':'normal'}">${event.alert?'Alert score':'Below threshold'}</td></tr>`).join('')}</tbody></table>`;
  }
  if(data.agents.length){
    $('live-agent-feed').innerHTML=data.agents.slice(-18).reverse().map(agent=>`<article class="live-agent-event"><span class="agent-role">${esc(agent.role)}</span><span class="agent-priority ${esc(agent.priority)}">${esc(agent.priority)}</span><p>${esc(agent.message)}</p><small>flow update ${agent.flow_id} · local rule agent</small></article>`).join('');
  }
}
async function pollLive(){
  try{renderLive(await get(`/api/live/events?after=${liveAfter}`));captureErrorShown=false}
  catch(error){if(!captureErrorShown){toast(`Live capture status unavailable: ${error.message}`);captureErrorShown=true}}
}
async function startCapture(){
  try{const interfaceName=$('capture-interface').value;if(!interfaceName)throw Error('Choose an available interface');await post('/api/live/start',{interface:interfaceName});liveAfter=0;ratePoints=[];await pollLive();activate('traffic')}
  catch(error){toast(`Capture could not start: ${error.message}`);await pollLive()}
}
async function stopCapture(){try{await post('/api/live/stop',{});await pollLive()}catch(error){toast(`Capture stop failed: ${error.message}`)}}
async function compareDomains(useScenario){
  try{
    const params=new URLSearchParams({feature:$('domain-feature').value});
    if(useScenario){params.set('load',$('domain-load').value);params.set('alerts',$('domain-alerts').value)}
    domainComparison=await get(`/api/domain/compare?${params}`);renderDomainComparison(domainComparison);
  }catch(error){toast(`Domain comparison failed: ${error.message}`)}
}
function renderDomainComparison(data){
  const source=data.source_domain,target=data.target_domain;
  $('domain-status').textContent=data.target_context_mode==='user_selected_context_scenario'?`Controlled context scenario ${data.target_context.join('/')} applied to the real B3 checkpoint. It is a selected operating condition, not an observed domain label.`:data.same_context?`Measured UNSW and CSE domains both map to context ${data.source_context.join('/')}. Correctly, the hypernetwork makes no parameter change for identical context inputs.`:`Measured domains map to different contexts ${data.source_context.join('/')} → ${data.target_context.join('/')}. Generated weights below come from the same trained checkpoint.`;
  $('domain-cards').innerHTML=card('UNSW measured context',data.source_context.join('/'),`${source.rows.toLocaleString()} A-train rows · alert ${pct(source.predicted_alert_rate)}`)+card('CSE measured context',target.measured_context.join('/'),`${target.rows.toLocaleString()} B-evaluation rows · alert ${pct(target.predicted_alert_rate)}`)+card('Generated context',data.target_context.join('/'),data.target_context_mode==='measured_domain'?'measured target domain':'selected scenario')+card('Parameter Δ · L2',fmt(data.parameter_delta_l2,6),`relative ${pct(data.parameter_delta_relative)}`)+card('Gate changes',data.changed_gate_count,`${data.source_neurons} → ${data.target_neurons} active neurons`)+card('Generated MACs',`${data.source_macs.toLocaleString()} → ${data.target_macs.toLocaleString()}`,'same network input dimension')+card('CSE B detection',`${pct(data.source_b_metrics.detection_rate)} → ${pct(data.target_b_metrics.detection_rate)}`,'offline labeled evaluation · A-val thresholds');
  $('domain-equations').textContent=`λ = [${data.preference.map(value=>Number(value).toFixed(4)).join(', ')}]\nsource context cₛ = [${data.source_context.join(', ')}]\ntarget context cₜ = [${data.target_context.join(', ')}]\nload bin: ${data.context_rules.load}\nalert bin: ${data.context_rules.alerts}\nγ = ${data.gate_constants.gamma}, ζ = ${data.gate_constants.zeta}, τ = ${data.gate_constants.temperature}\n\n${data.math.condition}\n${data.math.generator}\n${data.math.gate}\n${data.math.neuron}\n${data.math.network}\n\nActual generated parameter Δ₂ = ${data.parameter_delta_l2.toPrecision(7)}\nRelative parameter change = ${pct(data.parameter_delta_relative)}\nGate-vector Δ₂ = ${data.gate_delta_l2.toPrecision(7)}`;
  $('domain-neurons').innerHTML=data.neurons.map(neuron=>`<div class="domain-neuron"><strong>Neuron ${neuron.neuron}</strong><small>gate ${fmt(neuron.source_gate,3)} → ${fmt(neuron.target_gate,3)}</small><small>contribution Δ₂ ${fmt(neuron.contribution_delta_l2,5)}</small><small>output weight ${fmt(neuron.source_output_weight,4)} → ${fmt(neuron.target_output_weight,4)}</small></div>`).join('');
  drawDomainChart();
}
function drawDomainChart(){
  const data=domainComparison,canvas=$('domain-chart');if(!data||!canvas)return;
  const rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1,width=Math.floor(rect.width),height=Math.round(rect.height);
  if(width<120||height<120)return;
  canvas.width=Math.max(1,Math.floor(width*ratio));canvas.height=Math.floor(height*ratio);
  const context=canvas.getContext('2d');context.scale(ratio,ratio);context.clearRect(0,0,width,height);
  const left=58,right=18,top=20,bottom=38,all=[...data.source_logit_curve,...data.target_logit_curve],low=Math.min(...all),high=Math.max(...all),span=Math.max(high-low,1e-9);
  context.font='11px system-ui';context.fillStyle='#777';context.strokeStyle='#e8e8e8';context.lineWidth=1;
  for(let tick=0;tick<5;tick++){const y=top+tick*(height-top-bottom)/4;context.beginPath();context.moveTo(left,y);context.lineTo(width-right,y);context.stroke();context.fillText((high-tick*span/4).toFixed(3),4,y+4)}
  const draw=(values,color,dash)=>{context.beginPath();context.strokeStyle=color;context.lineWidth=2;context.setLineDash(dash);values.forEach((value,index)=>{const x=left+index*(width-left-right)/(values.length-1),y=top+(high-value)/span*(height-top-bottom);if(index===0)context.moveTo(x,y);else context.lineTo(x,y)});context.stroke();context.setLineDash([])};
  draw(data.source_logit_curve,'#171717',[]);draw(data.target_logit_curve,'#777',[5,4]);
  context.fillStyle='#555';context.fillText(`${data.feature} · real CSE B percentiles`,left,height-8);
  canvas.onmousemove=event=>{const bounds=canvas.getBoundingClientRect(),plotWidth=Math.max(1,bounds.width-left-right),ratioX=Math.max(0,Math.min(1,(event.clientX-bounds.left-left)/plotWidth)),index=Math.max(0,Math.min(data.feature_points.length-1,Math.round(ratioX*(data.feature_points.length-1))));$('domain-hover').textContent=`${data.feature} = ${Number(data.feature_points[index]).toPrecision(5)} · source output ${Number(data.source_logit_curve[index]).toFixed(5)} · target output ${Number(data.target_logit_curve[index]).toFixed(5)}`};
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
  $('model-cards').innerHTML=card('B detection · int8',pct(data.int8.detection_rate),'recorded CSE flow rows')+card('B false positives',pct(data.int8.fpr),'A-val calibrated threshold')+card('Float/int8 alert agreement',pct(data.alert_decision_agreement),`${data.evaluation_rows.toLocaleString()} recorded rows`)+card('Model cost',data.macs.toLocaleString(),'MACs / flow')+card('Kernel latency',`${fmt(data.latency.median_ns_per_flow,1)} ns`,'unpinned kernel-only timing')+card('Model size',`${data.int8_model_bytes} B`,'generated int8 artifact');
  $('neurons').innerHTML=data.gate_mask.map((open,i)=>`<span class="neuron ${open?'':'off'}" title="Neuron ${i+1}: ${open?'active':'pruned'}">${i+1}</span>`).join('');
  renderExportProof(data);
  $('model-note').textContent=`Offline checkpoint: ${data.source_run_id}. Generated registry event ${data.registry_event_id}; ${data.evaluation_rows.toLocaleString()} recorded NF-v2 flow rows, no live packets. Context ${data.context.join('/')}. The slider generates and bakes from the trained checkpoint, without optimizer updates. Float/int8 detection-rate difference at separately A-val-calibrated thresholds: ${pct(data.int8_detection_gap)}; resulting alert decisions agree on ${pct(data.alert_decision_agreement)} of rows. ${data.timing_note}.`;
  $('traffic-summary').innerHTML=card('B detection · int8',pct(data.int8.detection_rate),'selected generated candidate')+card('B FPR',pct(data.int8.fpr),'fixed A-val threshold')+card('B PR-AUC',fmt(data.int8.pr_auc),'attack ranking');
}

function renderExportProof(data){
  const artifact=data.export,check=data.kernel_verification,q=artifact.quantization;
  const pairs=check.sample_integer_logits.map(row=>`<tr><td>${row.row}</td><td>${row.c}</td><td>${row.python}</td><td>${row.c===row.python?'Exact':'Mismatch'}</td></tr>`).join('');
  const modelUrl=`/api/artifact?name=${encodeURIComponent(artifact.filename)}`;
  const manifestUrl=`/api/artifact?name=${encodeURIComponent(artifact.manifest_filename)}`;
  $('export-proof').innerHTML=`<div class="panel export-proof"><div class="panel-head"><div><p class="eyebrow">GENERATED ARTIFACT · KERNEL PARITY</p><h3>Export and inference verification</h3></div><span class="export-pass">${check.verified?'Exact match':'Verification failed'}</span></div><div class="cards">${card('Pruned detector',`${artifact.active_neurons} / ${artifact.total_neurons} active`,`${artifact.pruned_neurons} gated neurons removed`)}${card('Export dimensions',`${artifact.dimensions.neurons_after_pruning} × ${artifact.dimensions.hidden_per_neuron} × ${artifact.dimensions.input_features}`,'neurons × hidden × input features')}${card('C vs Python',`${check.rows_compared.toLocaleString()} rows`,`${check.mismatched_rows} mismatches · max integer delta ${check.maximum_absolute_logit_difference}`)}${card('Quantized weights',`${q.weight_value_count.toLocaleString()} values`,`${q.weight_values_at_int8_limit.toLocaleString()} at ±127`)}${card('Export size',`${data.int8_model_bytes.toLocaleString()} bytes`,'HWF2 int8 model')}${card('Reproducibility',data.optimizer_steps_during_generation===0?'No training step':'Training occurred','generated from checkpoint')}</div><div class="export-files"><div><small>Model SHA-256</small><code>${esc(artifact.sha256)}</code></div><div><small>Manifest SHA-256</small><code>${esc(artifact.manifest_sha256)}</code></div></div><div class="export-downloads"><a class="button-link primary-link" href="${modelUrl}">Download .hwi detector</a><a class="button-link" href="${manifestUrl}">Download export manifest</a></div><details><summary>Quantization scales</summary><pre>${esc(JSON.stringify(q.scales,null,2))}</pre></details><div class="table-wrap"><table><thead><tr><th>B evaluation row</th><th>C int32 logit</th><th>Python int32 logit</th><th>Parity</th></tr></thead><tbody>${pairs}</tbody></table></div><p class="muted">Full paired integer-logit comparison was performed on all listed rows against the exported file. Exact means zero mismatches, not merely similar floating-point scores.</p></div>`;
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
    const checks=Object.entries(data.constraint_checks||{}).map(([name,ok])=>`<span class="check ${ok?'ok':'fail'}">${ok?'✓':'×'} ${esc(name.replaceAll('_',' '))}</span>`).join('');
    const max=Math.max(1,...(data.feature_changes||[]).map(x=>x.mean_relative_delta_pct));
    const bars=(data.feature_changes||[]).map(x=>`<div class="delta-row"><span>${esc(x.feature)}</span><div class="delta-track"><i style="width:${Math.max(1,100*x.mean_relative_delta_pct/max)}%"></i></div><b>${fmt(x.mean_relative_delta_pct,2)}%</b><small>${x.changed_flows.toLocaleString()} flows</small></div>`).join('');
    const examples=(data.examples||[]).map(row=>`<tr><td>Flow ${row.flow}</td><td>${fmt(row.original_score,4)}</td><td>${fmt(row.candidate_score,4)}</td><td>${row.original_alert?'Alert':'Pass'}</td><td>${row.candidate_alert?'Alert':'Pass'}</td><td>${row.changed_features}</td></tr>`).join('');
    $('evasion-result').innerHTML=`<div class="cards">${card('Attack flows scored',data.attack_rows.toLocaleString(),`${esc(data.evaluation_partition)} · paired rows`)}${card('Detection before',pct(data.original_detection_rate),`${data.original_detected.toLocaleString()} above threshold`)}${card('Detection after',pct(data.evasion_detection_rate),`${data.candidate_detected.toLocaleString()} above same threshold`)}${card('Detected → missed',data.detected_to_missed.toLocaleString(),'paired decision flips')}${card('Flows modified',data.changed_flows.toLocaleString(),`${data.changed_feature_count} features changed`)}${card('Score threshold',fmt(data.threshold,4),data.threshold_source)}</div>
      <div class="panel"><div class="panel-head"><h3>Measured feature deltas</h3><span>Mean absolute change ÷ max(|original|, 1)</span></div>${bars||'<p class="muted">No feature values changed at this budget.</p>'}</div>
      <div class="panel"><div class="panel-head"><h3>Constraint audit</h3><span>${esc(data.kind.replaceAll('_',' '))} · ε ${fmt(data.epsilon,2)} · seed ${data.perturbation_seed}</span></div><div class="checks">${checks}</div></div>
      <div class="panel"><div class="panel-head"><h3>Paired flow examples</h3><span>First 12 modified rows · scores from same model and threshold</span></div><div class="table-wrap"><table><thead><tr><th>Evaluation row</th><th>Original score</th><th>Changed score</th><th>Before</th><th>After</th><th>Changed features</th></tr></thead><tbody>${examples||'<tr><td colspan="6">No modified flows in this run.</td></tr>'}</tbody></table></div><p class="muted">${esc(data.scope_note)}</p></div>`;
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
document.querySelectorAll('.map-node').forEach(node=>node.addEventListener('click',()=>selectMapNode(node)));
selectMapNode(document.querySelector('.map-node'));
$('limits-button').addEventListener('click',()=>$('limits').showModal());$('close-limits').addEventListener('click',()=>$('limits').close());
let timer;$('cost').addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(generate,220)});
$('load').addEventListener('change',generate);$('alerts').addEventListener('change',generate);
$('autopilot').addEventListener('click',autopilot);
$('run-evasion').addEventListener('click',evasion);$('approve').addEventListener('click',()=>decide(true));$('reject').addEventListener('click',()=>decide(false));
$('capture-start').addEventListener('click',startCapture);$('capture-stop').addEventListener('click',stopCapture);
window.addEventListener('resize',drawRateChart);
$('compare-domains').addEventListener('click',()=>compareDomains(false));$('apply-domain-context').addEventListener('click',()=>compareDomains(true));
(window.addEventListener('resize',drawDomainChart));
(async()=>{try{overview=await get('/api/overview');renderOverview();await loadCaptureInterfaces();await generate();await pollLive();livePollTimer=setInterval(pollLive,700)}catch(e){toast(e.message)}})();
