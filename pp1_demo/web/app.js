'use strict';
const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, x => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const fmt = n => Number(n).toFixed(1);
const clock = n => `${Math.floor(n/60)}:${String(Math.floor(n%60)).padStart(2,'0')}`;
const pct = n => `${Math.round(Number(n)*100)}%`;
const prettyChord = s => String(s).replace(':maj','').replace(':min','m').replace(':hdim7','ø7').replace(':dim','dim').replace(':aug','+').replace(':sus4','sus4').replace(':','');
// Same relative-root and family vocabularies as the Temporal Transformer notebook.
const romanRoots = ['I','♭II','II','♭III','III','IV','♯IV','V','♭VI','VI','♭VII','VII'];
const chordFamilies = ['MAJ','MIN','DOM','DIM','HDIM','AUG','SUS'];
const romanChord = c => c && Number.isInteger(c.relative_root) && romanRoots[c.relative_root] && chordFamilies.includes(c.family) ? `${romanRoots[c.relative_root]} · ${c.family}` : '—';
const stageInfo = [
  {id:'praveen', number:'01', title:'Vocal assessment', owner:'ExtraTrees', glyph:'∿', input:'Vocal waveform', output:'Accepted audio + repair proposal'},
  {id:'vishwa', number:'02', title:'Melody understanding', owner:'pYIN + CNN / Transformer', glyph:'♬', input:'BPDR accepted waveform', output:'Live notes + learned pitch classes'},
  {id:'pasan', number:'03', title:'Harmonic planning', owner:'Temporal Transformer', glyph:'▥', input:'16-bar melody & beat context', output:'Roman root + chord family'},
  {id:'rahul', number:'04', title:'Chord refinement', owner:'Multi-task Transformer', glyph:'✧', input:'Basic chords + musical context', output:'Refined chords + top candidates'}
];
let researchPanels={};
let tracks=[], genres=[], selected=null, run=null, activeStage='vishwa', poll=null, timer=null, lastFingerprint='', listenFingerprint='', inspectPinned=false, busy=false, startedAt=0, selectionRequest=0;
async function request(url, options) {
  const response = await fetch(url,options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'The local service could not complete this request.');
  return data;
}
function toast(message){$('toast').textContent=message;$('toast').hidden=false;setTimeout(()=>$('toast').hidden=true,6000);}
function tag(text,saved=false){return `<span class="provenance-tag ${saved?'saved':''}">${escapeHtml(text)}</span>`;}
function metric(value,label){return `<div class="mini-metric"><strong>${escapeHtml(value)}</strong><small>${escapeHtml(label)}</small></div>`;}
function io(inputTitle,inputText,outputTitle,outputText){return `<div class="io-grid"><div class="io-box"><small>INPUT</small><h4>${escapeHtml(inputTitle)}</h4><p>${escapeHtml(inputText)}</p></div><div class="io-arrow">→</div><div class="io-box"><small>OUTPUT</small><h4>${escapeHtml(outputTitle)}</h4><p>${escapeHtml(outputText)}</p></div></div>`;}
function inspectionData(data){
  if(Array.isArray(data))return data.map(inspectionData);
  if(data&&typeof data==='object')return Object.fromEntries(Object.entries(data).map(([key,value])=>[key.replace('rahul','refinement').replace('pasan','harmonic_planning').replace('praveen','vocal_assessment').replace('vishwa','feature_extraction'),key==='checkpoint'?String(value).split(/[\\/]/).at(-1):inspectionData(value)]));
  if(typeof data==='string')return data.replaceAll('pasan_model','harmonic_planning_model').replaceAll('rahul_model','chord_refinement_model').replace(/Vishwa/g,'Feature extraction');
  return data;
}
function details(title,data){return `<div class="detail-footer"><details><summary>${escapeHtml(title)}</summary><pre class="json">${escapeHtml(JSON.stringify(inspectionData(data),null,2))}</pre></details></div>`;}
function header(title,body,provenance,saved=false){return `<div class="detail-header"><div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(body)}</p></div>${tag(provenance,saved)}</div>`;}
function empty(title,body){return `<div class="empty-chart"><span class="empty-symbol">∿</span><p>${escapeHtml(title)}</p><small>${escapeHtml(body)}</small></div>`;}
function axes(duration,width=920,height=180){let out='';for(let i=0;i<=5;i++){let x=42+(width-58)*i/5;out+=`<line class="axis" x1="${x}" y1="9" x2="${x}" y2="${height-24}"/><text x="${x}" y="${height-6}" text-anchor="middle">${fmt(duration*i/5)}s</text>`;}return out;}
function pianoRoll(notes,beats,duration){
  const W=920,H=190,L=42,R=16,B=25,T=10;
  let low=Math.min(...notes.map(n=>n.pitch_midi),60)-2,high=Math.max(...notes.map(n=>n.pitch_midi),72)+2;
  if(high-low<12){low-=4;high+=4;}
  const x=t=>L+(W-L-R)*t/duration, y=p=>T+(H-B-T)*(high-p)/(high-low);
  let body=axes(duration,W,H);
  for(let p=Math.ceil(low/12)*12;p<high;p+=12){body+=`<line class="axis" x1="${L}" x2="${W-R}" y1="${y(p)}" y2="${y(p)}"/><text x="29" y="${y(p)+3}" text-anchor="end">C${Math.floor(p/12)-1}</text>`;}
  for(const b of beats){const start=Math.max(0,b.start_s);body+=`<line x1="${x(start)}" x2="${x(start)}" y1="${T}" y2="${H-B}" stroke="${b.beat_position===1?'#688965':'#37503e'}" opacity="${b.beat_position===1?.6:.35}" stroke-dasharray="${b.beat_position===1?'3 3':'1 4'}"/>`;}
  for(const n of notes){body+=`<rect x="${x(n.onset_s)}" y="${y(n.pitch_midi)-3}" width="${Math.max(2,x(n.onset_s+n.duration_s)-x(n.onset_s))}" height="6" rx="2" fill="#b0ecb5" opacity=".85"><title>MIDI ${n.pitch_midi} · ${fmt(n.onset_s)}–${fmt(n.onset_s+n.duration_s)}s · ${escapeHtml(n.source||'saved extraction')}</title></rect>`;}
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Melody piano roll with beat positions">${body}</svg>`;
}
function lineChart(series,duration,label,color='#b0ecb5'){
  const W=920,H=175,L=42,R=16,T=14,B=26;
  const valid=series.filter(p=>p[1]!==null&&Number.isFinite(p[1]));
  if(!valid.length)return empty('No eligible measurements','The original audio remains available.');
  let lo=Math.min(...valid.map(p=>p[1])), hi=Math.max(...valid.map(p=>p[1]));
  if(hi-lo<1){lo-=1;hi+=1;}else{const pad=(hi-lo)*.1;lo-=pad;hi+=pad;}
  const x=t=>L+(W-L-R)*t/duration,y=v=>T+(H-T-B)*(hi-v)/(hi-lo);
  let path='',gap=true;
  for(const p of series){if(p[1]===null||!Number.isFinite(p[1])){gap=true;continue;}path+=`${gap?'M':'L'}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)} `;gap=false;}
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHtml(label)}">${axes(duration,W,H)}<text x="31" y="${T+4}" text-anchor="end">${Math.round(hi)}</text><text x="31" y="${H-B}" text-anchor="end">${Math.round(lo)}</text><path d="${path}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round"/></svg>`;
}
const chordPalette=['#9cccaa','#7bbda7','#83aabd','#acb591','#c7b28a','#a8a2d2','#78afab'];
function chordTimeline(rows,duration,label='Chords',isRefined=false){
  const W=920,H=110,L=42,R=16,usable=W-L-R;
  let body=axes(duration,W,H);
  let groups=[];
  for(const row of rows){const last=groups.at(-1);if(last&&last.label===row.label&&Math.abs(last.end_s-row.start_s)<.001){last.end_s=row.end_s;}else groups.push({...row});}
  groups.forEach((c,i)=>{const x=L+c.start_s/duration*usable,w=Math.max(1,(c.end_s-c.start_s)/duration*usable),title=prettyChord(c.label),fill=isRefined?['#a99bd6','#b9a9d8','#91a5c1','#c0b5e0'][i%4]:chordPalette[i%chordPalette.length];body+=`<rect x="${x}" y="24" width="${Math.max(1,w-1)}" height="39" rx="4" fill="${fill}" opacity=".85"><title>${escapeHtml(c.label)} · ${fmt(c.start_s)}–${fmt(c.end_s)}s</title></rect>`;if(w>24)body+=`<text x="${x+w/2}" y="48" text-anchor="middle" style="fill:#14261b;font-size:${w>50?11:8}px;font-weight:600">${escapeHtml(title)}</text>`;});
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHtml(label)}">${body}</svg>`;
}
function romanTimeline(rows,duration,label='Predicted Roman roots and chord families'){
  return chordTimeline(rows.map(c=>({...c,label:romanChord(c)})),duration,label);
}
function melodyInput(){
  if(!selected)return '';
  const notes=run?.context?.notes||selected.notes,beats=run?.context?.beats||selected.beats;
  return `<div class="chart-box input-melody"><div class="chart-title"><span>Input melody · piano roll</span><small>${run?.vishwa?'Live pYIN inside the crop':'Saved preview before inference'} · dashed lines mark saved beats</small></div>${pianoRoll(notes,beats,selected.duration_s)}<p class="small muted">The audible excerpt is shown here. Inference uses ${run?.context?.context_notes||selected.context_notes} notes and ${selected.context_beats} beat queries across 16 complete bars of the same track.</p></div>`;
}
function familyLegend(){
  return `<div class="family-vocabulary"><p class="small muted">Seven chord-family classes</p><div class="pill-row">${chordFamilies.map(f=>`<span class="pill">${f}</span>`).join('')}</div></div>`;
}
function confidenceChart(rows,duration){
  const W=920,H=130,L=42,U=W-L-16;
  let body=axes(duration,W,H);
  rows.forEach(c=>{const x=L+c.start_s/duration*U,w=Math.max(1,(c.end_s-c.start_s)/duration*U);body+=`<rect x="${x}" y="${100-c.root_probability*80}" width="${Math.max(1,w-2)}" height="${c.root_probability*80}" rx="2" fill="#89b79a" opacity=".8"><title>Root: ${pct(c.root_probability)} · Family: ${pct(c.family_probability)}</title></rect>`;});
  body+='<text x="30" y="25" text-anchor="end">100%</text><text x="30" y="100" text-anchor="end">0%</text>';
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Uncalibrated root softmax scores per beat">${body}</svg>`;
}
function drawWaveform(){
  if(!selected)return;
  const values=selected.waveform,W=1100,H=80,step=W/values.length;
  const max=Math.max(...values,.00001);let body='';
  values.forEach((v,i)=>{let h=Math.max(2,Math.pow(v/max,.65)*H);body+=`<rect x="${i*step}" y="${(H-h)/2}" width="${Math.max(.8,step-1.3)}" height="${h}" rx="1" fill="#82aa8d" data-wave-index="${i}"/>`;});
  $('waveform').innerHTML=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Waveform of the selected vocal excerpt">${body}</svg>`;
}
function drawLibrary(){
  const query=($('library-search').value||'').trim().toLowerCase();
  const visible=tracks.filter(t=>`${t.id} ${t.title} ${t.genres.join(' ')} ${t.key.tonic} ${t.key.mode}`.toLowerCase().includes(query));
  $('track-count').textContent=query?`${visible.length}/${tracks.length}`:String(tracks.length);
  $('track-list').innerHTML=visible.map(t=>`<button class="track-button ${t.id===selected?.id?'selected':''}" data-track="${escapeHtml(t.id)}" aria-pressed="${t.id===selected?.id}" ${busy?'disabled':''}><span class="track-art">∿</span><span><span class="track-name">${escapeHtml(t.title)}</span><span class="track-meta">${escapeHtml(t.genre)} · ${escapeHtml(t.key.tonic)} ${escapeHtml(t.key.mode)}</span></span><span class="track-time">${clock(t.duration_s)}</span></button>`).join('') || '<p class="small muted">No matching recordings.</p>';
  document.querySelectorAll('[data-track]').forEach(b=>b.onclick=()=>selectTrack(b.dataset.track));
}
async function selectTrack(id){
  if(busy||!tracks.some(t=>t.id===id))return;
  const ticket=++selectionRequest;
  $('run-button').disabled=true;
  let preview;
  try{preview=await request('/api/tracks/'+encodeURIComponent(id));}
  catch(error){if(ticket===selectionRequest){$('run-button').disabled=!selected;toast(error.message);}return;}
  if(ticket!==selectionRequest)return;
  selected=preview;
  run=null;lastFingerprint='';listenFingerprint='';activeStage='vishwa';inspectPinned=false;
  $('track-title').textContent=selected.title;
  $('track-subtitle').textContent=`${fmt(selected.duration_s)}-second vocal · excerpt ${fmt(selected.crop_offset_s)}–${fmt(selected.crop_offset_s+selected.duration_s)}s in the original song`;
  $('source-audio').src=selected.audio_url;
  $('source-facts').innerHTML=`<div><span class="fact-label">Reference key</span><span class="fact-value">${escapeHtml(selected.reference_key)}</span></div><div><span class="fact-label">Saved tempo</span><span class="fact-value">${selected.bpm} <small>BPM</small></span></div><div><span class="fact-label">Genre</span><span class="fact-value"><select id="genre-select" aria-label="Genre">${genres.filter(g=>g!=='UNK').map(g=>`<option ${g===selected.genre?'selected':''} value="${escapeHtml(g)}">${escapeHtml(g)}</option>`).join('')}</select></span></div><div><span class="fact-label">Inference context</span><span class="fact-value">16 <small>bars</small></span></div>`;
  drawWaveform();drawLibrary();renderFlow();renderTabs();renderStage();renderRunState();renderListening();
  $('run-button').disabled=false;
}
function stageStatus(id){return run?.stages?.[id]?.status||'pending';}
function renderFlow(){
  $('flow').innerHTML=stageInfo.map(s=>{const state=stageStatus(s.id),status=state==='complete'?'Complete':state==='running'?'Processing':state==='failed'?'Needs attention':s.id==='context'?'Saved features':'Ready';const count=run?.[s.id]?.chords?.length;return `<button class="flow-card ${state} ${activeStage===s.id?'chosen':''}" data-stage="${s.id}" aria-label="Inspect ${s.title}"><div class="flow-top"><span class="stage-glyph">${s.glyph}</span><span class="stage-num">${s.number}</span></div><h3>${s.title}</h3><p>${s.owner}</p><div class="stage-state"><span class="status">${status}</span><span>${count?count+' beats':state==='complete'?(run.stages[s.id].seconds+'s'):'↗'}</span></div></button>`;}).join('');
  document.querySelectorAll('.flow-card').forEach(b=>b.onclick=()=>chooseStage(b.dataset.stage));
}
function chooseStage(id){activeStage=id;inspectPinned=true;renderFlow();renderTabs();renderStage();$('inspection').scrollIntoView({behavior:'smooth',block:'start'});}
function renderTabs(){
  $('stage-tabs').innerHTML=stageInfo.map(s=>`<button class="stage-tab" id="tab-${s.id}" role="tab" aria-selected="${activeStage===s.id}" aria-controls="stage-detail" data-tab="${s.id}"><span>${s.number}</span>${s.title}</button>`).join('');
  document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{activeStage=b.dataset.tab;inspectPinned=true;renderFlow();renderTabs();renderStage();});
}
function featureHeatmap(values,times,labels,title,probability=false){
  if(!values.length||!times.length)return '';
  const W=920,H=220,L=42,T=12,B=25,U=W-L-16,V=H-T-B;
  const stride=Math.max(1,Math.ceil(times.length/120)),duration=selected.duration_s;
  let body=axes(duration,W,H);
  values.forEach((row,j)=>{for(let i=0;i<times.length;i+=stride){
    const value=Number(row[i]),fraction=Math.max(0,Math.min(1,probability?value:(value+80)/80));
    const x=L+times[i]/duration*U,w=Math.max(1,((times[Math.min(times.length-1,i+stride)]||duration)-times[i])/duration*U);
    body+=`<rect x="${x}" y="${T+V*(values.length-1-j)/values.length}" width="${w+1}" height="${V/values.length+1}" fill="hsl(${270-220*fraction},65%,${10+65*fraction}%)"><title>${escapeHtml(labels[j])} · ${fmt(times[i])}s · ${value.toFixed(3)}</title></rect>`;
  }});
  return `<div class="chart-box"><div class="chart-title"><span>${escapeHtml(title)}</span><small>${probability?'Rest then C…B, bottom to top · probability 0–1':'80 Mel bands low to high · −80 to 0 dB'}</small></div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHtml(title)}">${body}</svg></div>`;
}
function renderVishwa(){
  const v=run?.vishwa,state=stageStatus('vishwa');
  let html=header('Understand the melody from the accepted voice.','Live audio preparation, Mel features and pYIN notes; the trained CNN and CNN–Transformer run alongside them as weak-score pitch-class experiments.','LIVE AUDIO + CHECKPOINTS');
  html+=io('BPDR accepted waveform','Mono · 22,050 Hz · peak normalized · source timing preserved','Notes and parallel learned predictions','pYIN notes drive the harmonic models. CNN/hybrid predict rest or pitch class, without octave or note boundaries.');
  if(!v)html+=empty(state==='running'?'Extracting melody and running both checkpoints…':'Ready to analyse the accepted audio.','Saved library notes are only a pre-run preview. This stage must succeed before the downstream models run.');
  else{
    html+=`<div class="metrics-row">${metric(v.notes.length,'live note events')}${metric('18,413','CNN parameters')}${metric('31,117','hybrid parameters')}${metric('2','trained checkpoints executed')}</div>`;
    html+=`<div class="chart-box"><div class="chart-title"><span>Live melody notes</span><small>pYIN on accepted audio · original crop time</small></div>${pianoRoll(v.notes,run?.context?.beats||selected.beats,selected.duration_s)}</div>`;
    html+=`<div class="chart-box"><div class="chart-title"><span>Voiced fundamental frequency</span><small>Hz · silence masked</small></div>${lineChart(v.chart.times_s.map((t,i)=>[t,v.chart.f0_hz[i]]),selected.duration_s,'Live pYIN F0')}</div>`;
    html+=featureHeatmap(v.chart.mel_db,v.chart.times_s,v.chart.mel_db.map((_,i)=>'Mel band '+i),'Live Mel spectrogram');
    for(const [name,model] of Object.entries(v.pilot_models)){
      const columns=model.classes.map((_,i)=>model.chart_probabilities.map(row=>row[i]));
      html+=featureHeatmap(columns,v.chart.times_s,model.classes,name+' · actual checkpoint predictions',true);
    }
    html+=`<div class="notice amber">These classifiers have no octave or note-boundary head. Their predictions are shown honestly as experimental outputs. Live pYIN notes replace the entire audible crop, including rests; only surrounding notes remain saved. Beats/key are still saved inputs.</div>`;
    if(run.artifacts.vishwa_prepared)html+=`<div class="chart-box"><div class="chart-title"><span>Your prepared audio</span></div><audio controls src="${artifact(run.artifacts.vishwa_prepared)}"></audio></div>`;
    html+=details('Inspect checkpoint identities, preprocessing and handoff',{input_audio_sha256:v.input_audio_sha256,bundle_manifest_sha256:v.bundle_manifest_sha256,preprocessing:v.preprocessing,downstream_source:v.downstream_source,models:Object.fromEntries(Object.entries(v.pilot_models).map(([name,m])=>[name,{checkpoint_sha256:m.checkpoint_sha256,parameters:m.parameters,historical_validation_accuracy:m.historical_validation_recording_macro_accuracy}])),context:run.context,warnings:v.warnings});
  }
  for(const [name,text] of Object.entries(researchPanels))html+=`<div class="detail-footer"><details><summary>${escapeHtml(name)}</summary><pre style="white-space:pre-wrap;line-height:1.7">${escapeHtml(text)}</pre></details></div>`;
  return html;
}

function renderPraveen(){
  const p=run?.praveen,state=stageStatus('praveen');
  let html=header('Listen first. Assess the vocal.','The notebook-trained ExtraTrees model analyses local pitch context and proposes small corrections in cents. Its technical audio handoff and experimental repair preview are kept separate.','LIVE MODEL');
  html+=io('The selected vocal waveform',`${selected?fmt(selected.duration_s):'—'} seconds · mono vocal excerpt · sample integrity checked`,'Accepted audio + a correction proposal','The owner’s model and operating policy determine whether to leave audio unchanged or offer an experimental preview.');
  if(!p)return html+empty(state==='running'?'The vocal assessment model is analysing the recording…':'Ready for the first model','The real assessment, F0 contour and correction estimates appear after inference.');
  const ch=p.chart,accepted=p.handoff_status==='ACCEPT_UNCHANGED';
  html+=`<div class="notice ${p.reason?'amber':''}"><strong>${accepted?'Accepted · original pitch and timing preserved':escapeHtml(p.handoff_status)}</strong><br>${escapeHtml(p.message||p.forwarded_audio)}${p.reason?'<br>Repair preview: '+escapeHtml(p.reason):''}</div>`;
  if(run.artifacts.accepted)html+=`<div class="chart-box"><div class="chart-title"><span>Accepted audio · forwarded unchanged</span><small>Actual BPDR handoff</small></div><audio controls preload="metadata" src="${artifact(run.artifacts.accepted)}" aria-label="Play BPDR accepted audio"></audio>${run.artifacts.repair_preview?`<p class="small muted" style="margin:14px 0 8px">Separate experimental repair preview</p><audio controls src="${artifact(run.artifacts.repair_preview)}" aria-label="Play experimental BPDR repair preview"></audio>`:''}</div>`;
  html+=`<div class="metrics-row">${metric(ch.model_ran?((p.trees||96)+' trees'):'—','ExtraTrees inference')}${metric(pct(ch.voiced_fraction||0),'voiced acoustic frames')}${metric(ch.scope?.eligible_frames||0,'frames in held-note scope')}${metric('Unchanged','audio forwarded downstream')}</div>`;
  if(ch.times?.length){html+=`<div class="chart-box"><div class="chart-title"><span>Measured pitch contour</span><small>PESTO F0 · Hz · silence is masked</small></div>${lineChart(ch.times.map((t,i)=>[t,ch.f0_hz[i]]),selected.duration_s,'Measured performed pitch in Hz','#edba96')}</div>`;
  html+=`<div class="chart-box"><div class="chart-title"><span>Learned correction estimates</span><small>Raw model output in cents · proposal permissions applied separately</small></div>${lineChart(ch.times.map((t,i)=>[t,ch.raw_correction_cents[i]]),selected.duration_s,'ExtraTrees raw correction estimates in cents','#e8bd97')}</div>`;}
  html+=`<div class="notice amber">Technical acceptance means the recording can continue through the demo. It is not a “good singing” score. This repair model is experimental and requires review; its research/generalization gates did not pass.</div>`;
  html+=details('Inspect BPDR decisions and provenance',{outcome:p.outcome,checkpoint_sha256:p.checkpoint_sha256,scope:p.model_scope,decision:p.decision,forwarded_audio:p.forwarded_audio,requires_user_review:p.requires_user_review});
  return html;
}
function renderPasan(){
  const p=run?.pasan,state=stageStatus('pasan');
  let html=header('Plan harmony with temporal attention.','A melody encoder and beat-query decoder connect nearby notes to each chord position. The latest saved Temporal Transformer predicts a relative root and a coarse chord family.','LIVE MODEL');
  html+=io('Sixteen bars of melody + beats',`${selected?.context_notes||'—'} pitch/timing note events · ${selected?.context_beats||'—'} beat queries · C-based normalization`,'A Roman root + family at every beat','12 relative Roman roots × 7 chord families. Displayed relative to the reference key; the prediction handoff retains the root and family.');
  html+=melodyInput()+familyLegend();
  if(!p)return html+empty(state==='running'?'The Temporal Transformer is planning the progression…':'The first harmonic plan will appear here.','The checkpoint runs after vocal assessment and the live melody stage and context bridge complete.');
  html+=`<div class="metrics-row">${metric(p.metadata.epoch,'checkpoint epoch · latest')}${metric('16 bars','exact training context length')}${metric(p.chords.length,'chord queries in this excerpt')}${metric('2 beats','Gaussian attention width')}</div>`;
  html+=`<div class="chart-box"><div class="chart-title"><span>Predicted harmonic plan · Roman numerals</span><small>Relative to ${escapeHtml(selected.reference_key)} · Roman root + family</small></div>${romanTimeline(p.chords,selected.duration_s)}</div>`;
  html+=`<div class="chart-box"><div class="chart-title"><span>Root prediction scores</span><small>Raw softmax scores · not calibrated confidence</small></div>${confidenceChart(p.chords,selected.duration_s)}</div>`;
  html+=`<p class="small muted">${escapeHtml(p.bridge)} Genre is not an input to this model.</p>`;
  return html+details('Inspect model inputs, outputs and checkpoint',{...p.input_summary,root_vocabulary:romanRoots,family_vocabulary:chordFamilies,checkpoint:p.checkpoint,checkpoint_sha256:p.checkpoint_sha256,metadata:p.metadata,normalization_shift_semitones:p.normalization_shift_semitones,example_predictions:p.chords.slice(0,5).map(c=>({start_s:c.start_s,end_s:c.end_s,roman_root:romanRoots[c.relative_root],family:c.family,root_probability:c.root_probability,family_probability:c.family_probability}))});
}
function renderRahul(){
  const r=run?.rahul,state=stageStatus('rahul');
  let html=header('Refine the progression, beat by beat.','The multi-task Transformer combines the harmonic plan with melody, harmonic context, genre and timing. It predicts detailed chord labels and a separate auxiliary quality head.','LIVE MODEL');
  html+=io('Basic chords + original musical context','Live crop melody + saved outside context · key · Roman/function context derived from predicted chords · genre · mode fixed to major','Detailed chords + three candidates','Full-chord vocabulary: 402 classes · auxiliary quality vocabulary: 13 classes · four-bar inference groups.');
  html+=melodyInput();
  if(!r)return html+empty(state==='running'?'The chord refinement model is refining the harmonic plan…':'The refined progression appears after harmonic planning.','No reference chord targets are passed into either Transformer.');
  const p=run.pasan;
  const changes=r.chords.filter((c,i)=>c.label!==p.chords[i]?.label).length;
  html+=`<div class="metrics-row">${metric('402','full-chord output classes')}${metric('13','auxiliary quality classes')}${metric(r.chords.length,'refined beat predictions')}${metric('Major','explicit temporary mode')}</div>`;
  html+=`<div class="chart-box"><div class="chart-title"><span>Harmonic plan · Roman numerals</span><small>Input from harmonic planning · relative to ${escapeHtml(selected.reference_key)}</small></div>${romanTimeline(p.chords,selected.duration_s,'Input harmonic plan in Roman numerals')}<div class="chart-title"><span class="after-label">Refined progression · chord names</span><small>Extensions and inversions use the saved vocabulary</small></div>${chordTimeline(r.chords,selected.duration_s,'Refined chord progression',true)}</div>`;
  html+=`<div class="chart-box"><div class="chart-title"><span>Explore the predictions</span><small>First 12 beats · raw full-chord softmax probabilities</small></div><div class="chart-scroll"><table class="data-table"><thead><tr><th>Time</th><th>Harmonic plan input</th><th>Refined chord output</th><th>Top score</th><th>Alternatives</th></tr></thead><tbody>${r.chords.slice(0,12).map((c,i)=>`<tr><td>${fmt(c.start_s)}–${fmt(c.end_s)}s</td><td class="chord-label">${escapeHtml(romanChord(p.chords[i]))}</td><td class="chord-label after-label">${escapeHtml(prettyChord(c.label))}</td><td>${pct(c.probability)}<div class="confidence-line"><span style="width:${Math.min(100,c.probability*100)}%"></span></div></td><td class="candidate">${c.top3.slice(1).map(x=>escapeHtml(prettyChord(x.label))+' '+pct(x.probability)).join(' · ')}</td></tr>`).join('')}</tbody></table></div></div>`;
  html+=`<div class="notice blue">Mode is always <strong>major</strong> for this demonstration. Genre: <strong>${escapeHtml(run.settings.genre)}</strong> (${escapeHtml(run.provenance.genre.replaceAll('_',' '))}). The two prediction heads are separate; the auxiliary head is not used to re-score the displayed full chords.</div>`;
  const unknown=Object.entries(r.unknown_inputs||{});
  if(unknown.length)html+=`<p class="small muted">${unknown.length} input categories use the vocabulary’s UNK mapping. Exact details are retained in the handoff below.</p>`;
  return html+details('Inspect refinement inputs and results',{checkpoint:r.checkpoint,checkpoint_sha256:r.checkpoint_sha256,mode:r.mode,mode_source:r.mode_source,input_summary:r.input_summary,unknown_inputs:r.unknown_inputs,scores:r.scores,example_predictions:r.chords.slice(0,3)});
}
function renderStage(){
  if(!selected)return;
  $('stage-detail').setAttribute('role','tabpanel');$('stage-detail').setAttribute('aria-labelledby','tab-'+activeStage);
  $('stage-detail').innerHTML=({praveen:renderPraveen,context:renderVishwa,vishwa:renderVishwa,pasan:renderPasan,rahul:renderRahul}[activeStage]||renderVishwa)();
  const state=stageStatus(activeStage);
  $('inspect-state').textContent=state==='running'?'Live inference in progress':state==='complete'?'Actual result · '+run.stages[activeStage].seconds+'s':activeStage==='context'?'Mixed live / saved context':'Waiting for inference';
}
function renderRunState(){
  const active=run?.active_stage;
  busy=run?.status==='running'||run?.status==='queued';
  $('run-button').disabled=busy||!selected;
  $('run-button').innerHTML=busy?'<span>◌</span> Processing…':run?.status==='complete'?'<span>↻</span> Run again':'<span>▶</span> Run pipeline';
  if($('genre-select'))$('genre-select').disabled=busy;
  $('progress-fill').style.width=(run?.progress||0)+'%';
  $('run-dot').className='run-dot '+(busy?'running':'');
  const handoff=busy&&active&&run?.stages?.[active]?.status==='complete';
  $('run-status').textContent=handoff?({'vishwa':'Live melody ready · aligning with surrounding context…','context':'Live melody handoff ready · passing notes and saved beats to harmonic planning…','pasan':'Harmonic plan ready · passing predictions to chord refinement…','rahul':'Refined chords ready · preparing listening previews…'}[active]||'Handoff complete · continuing…'):run?.status==='complete'?'Pipeline complete · all four components ran':run?.status==='failed'?'This run stopped · inspect the error below':busy?({'praveen':'Assessing the vocal…','vishwa':'Extracting live melody and running both score checkpoints…','context':'Aligning live melody with saved musical context…','pasan':'Predicting the harmonic plan…','rahul':'Refining the chords…','render':'Rendering audio and MIDI previews…'}[active]||'Preparing the selected recording…'):'Ready · choose a recording and run the pipeline.';
  $('error-message').hidden=!run?.error;
  $('error-message').textContent=run?.error||'';
  if(run?.seconds)$('run-elapsed').textContent=run.seconds+'s total';
  if(run?.id)$('footer-run').textContent='RUN '+run.id.toUpperCase();
}
function artifact(name){return `/api/artifacts/${run.id}/${encodeURIComponent(name)}`;}
function renderListening(){
  const fingerprint=JSON.stringify([run?.id,run?.artifacts]);
  if(fingerprint===listenFingerprint)return;listenFingerprint=fingerprint;
  const a=run?.artifacts||{};
  if(!a.combined){$('downloads').innerHTML='';$('listen-players').innerHTML='<div class="card listen-placeholder"><span class="empty-symbol">♫</span><h3>Your harmony starts here.</h3><p>Run the pipeline to create melody, chord and combined audio previews.</p><small>Accompaniment is synthesized instrument audio.</small></div>';return;}
  const players=[['melody','LIVE MELODY OUTPUT','The extracted melody','Synthesized pYIN notes measured from BPDR accepted audio in this run.','♬'],['basic','HARMONIC PLANNING OUTPUT','The harmonic plan','An instrumental rendering of the root and family predictions.','▥'],['combined','VOICE + CHORD REFINEMENT OUTPUT','The finished demo','Accepted vocal mixed with synthesized, refined chord accompaniment.','✧']];
  $('listen-players').innerHTML=players.map(([key,kicker,title,body,icon],i)=>`<div class="card listen-card ${i===2?'featured':''}"><span class="listen-icon">${icon}</span><small>${kicker}</small><h3>${title}</h3><p>${body}</p><audio controls preload="metadata" src="${artifact(a[key])}" aria-label="Play ${escapeHtml(title)}"></audio></div>`).join('');
  $('downloads').innerHTML=`<a class="download" href="${artifact('manifest.json')}" download>↓ Run JSON</a><a class="download" href="${artifact(a.refined_midi)}" download>↓ Refined MIDI</a><a class="download" href="${artifact(a.combined)}" download>↓ Combined WAV</a>`;
}
async function pollRun(id){
  try{
    const next=await request('/api/runs/'+id);run=next;
    const fingerprint=JSON.stringify(next);
    if(fingerprint!==lastFingerprint){
      lastFingerprint=fingerprint;
      if(!inspectPinned&&next.active_stage&&next.active_stage!=='render')activeStage=next.active_stage==='context'?'vishwa':next.active_stage;
      if(!inspectPinned&&next.status==='complete')activeStage='rahul';
      renderFlow();renderTabs();renderStage();renderRunState();renderListening();drawLibrary();
    }
    if(['complete','failed'].includes(next.status)){
      clearInterval(poll);poll=null;clearInterval(timer);timer=null;
      if(next.status==='complete'){localStorage.setItem('harmonyLabRun',id);toast('All four components finished. Your listening previews are ready.');}
      return;
    }
  }catch(error){clearInterval(poll);poll=null;clearInterval(timer);timer=null;busy=false;if(run){run.status='failed';run.error='Could not read run progress: '+error.message;}renderRunState();toast('Could not read this run: '+error.message);}
}
async function startRun(){
  if(busy||!selected)return;
  busy=true;inspectPinned=false;activeStage='praveen';$('run-button').disabled=true;document.querySelector('.pipeline-section').scrollIntoView({behavior:'smooth',block:'start'});
  const genre=$('genre-select').value;
  try{
    const result=await request('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({track_id:selected.id,genre})});
    startedAt=Date.now();listenFingerprint='';
    if(poll)clearInterval(poll);if(timer)clearInterval(timer);
    timer=setInterval(()=>{if(busy)$('run-elapsed').textContent=Math.floor((Date.now()-startedAt)/1000)+'s elapsed';},1000);
    await pollRun(result.id);
    if(!['complete','failed'].includes(run?.status))poll=setInterval(()=>pollRun(result.id),900);
  }catch(error){busy=false;$('run-button').disabled=false;toast(error.message);}
}
function sourceDetails(){
  if(!selected)return;
  $('source-dialog-body').innerHTML=`<p><strong>${escapeHtml(selected.title)}</strong> is an existing vocal excerpt from your local MTG-Jamendo extraction outputs. It has not been newly recorded or generated by this demo.</p><div class="metrics-row">${metric(fmt(selected.duration_s)+'s','playable vocal excerpt')}${metric(fmt(selected.crop_offset_s)+'s','offset in original song')}</div><p>The source is an existing WAV vocal crop. Its saved checksum is checked before processing. The UI uses the source format directly rather than converting it to MP3.</p><p>Melody is extracted live from accepted audio inside this crop. Saved melody is retained only outside the crop; beats and key remain saved metadata. The two Transformers use ${fmt(selected.context_duration_s)} seconds of complete sixteen-bar context from the same track, then crop their output timelines back to the audible excerpt.</p><p>Source genre tags: <strong>${escapeHtml(selected.genres.join(', ')||'none')}</strong>. Original saved key estimate: <strong>${escapeHtml(selected.key.tonic+' '+selected.key.mode)}</strong>. Chord refinement mode is independently fixed to <strong>major</strong>.</p><p>No reference chord labels are used as inference inputs. Instrument previews are synthesized accompaniment, not generated harmony vocals.</p>`;
  $('source-dialog').showModal();
}
$('library-search').addEventListener('input',drawLibrary);
$('run-button').onclick=startRun;$('source-details-button').onclick=sourceDetails;$('close-dialog').onclick=()=>$('source-dialog').close();
$('source-dialog').addEventListener('click',event=>{if(event.target===$('source-dialog'))$('source-dialog').close();});
document.querySelectorAll('.owner-nav').forEach(b=>b.onclick=()=>chooseStage(b.dataset.stage));
$('source-audio').addEventListener('timeupdate',()=>{const fraction=$('source-audio').currentTime/(selected?.duration_s||1);document.querySelectorAll('[data-wave-index]').forEach(rect=>{rect.setAttribute('fill',Number(rect.dataset.waveIndex)/(selected?.waveform.length||1)<fraction?'#c7f3b8':'#82aa8d');});});
$('waveform').onclick=event=>{if(!selected)return;const box=$('waveform').getBoundingClientRect();$('source-audio').currentTime=Math.max(0,Math.min(selected.duration_s,(event.clientX-box.left)/box.width*selected.duration_s));};
document.addEventListener('play',event=>{if(event.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==event.target)a.pause();});},true);
(async function init(){
  try{
    const data=await request('/api/catalog');tracks=data.tracks;genres=data.genres;researchPanels=await request('/api/panels/vishwa');
    $('connection').textContent=Object.values(data.models).every(Boolean)?'Four components connected':'Some model files are missing';
    drawLibrary();await selectTrack(tracks[0].id);
  }catch(error){$('connection').textContent='Local assets unavailable';$('track-title').textContent='The library could not load';$('track-subtitle').textContent=error.message;toast(error.message);}
})();
