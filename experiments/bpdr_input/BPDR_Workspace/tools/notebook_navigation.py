import html

def navigator(navigation,receipt,questions):
    esc=html.escape
    links={r['id']:r['html'] for r in navigation['blocks']}
    links.update({'RESEARCH-STATUS':'01_research_foundations.html#RESEARCH-STATUS',
                  'HANDOFF-CONTRACT':'05_audio_processing_and_handoff.html#HANDOFF-CONTRACT',
                  'LITERATURE':'01_research_foundations.html#LITERATURE'})
    cards=[]
    for row in navigation['blocks']:
        if row['id'].endswith(('-SETUP','-CONTEXT')):continue
        search=esc((row['id']+' '+row['title']+' '+row['notebook']).lower(),quote=True)
        cards.append(f'''<article class="card" data-kind="blocks" data-search="{search}" id="{row['id']}">
        <div class="eyebrow">{esc(row['id'])} · notebook {row['notebook'][:2]} / cell {row['cell']}</div>
        <h3>{esc(row['title'])}</h3><p class="path">{esc(row['notebook'])}</p>
        <a class="action" href="{esc(row['html'])}">Open code &amp; executed output ↗</a></article>''')
    for row in navigation['sources']:
        search=esc((' '.join([row['id'],row['title'],row['path'],row['function'],row['target']])).lower(),quote=True)
        cards.append(f'''<article class="card source" data-kind="sources" data-search="{search}" id="{row['id']}">
        <div class="eyebrow">{esc(row['id'])} · source function</div><h3>{esc(row['title'])}</h3>
        <p class="path">{esc(row['path'])}:{row['line']} · {esc(row['function'])}</p>
        <p><a href="{esc(links[row['target']])}">Walkthrough / {esc(row['target'])} ↗</a></p>
        <details><summary>Show implementation · lines {row['line']}–{row['end_line']}</summary>
        <button type="button" class="copy" data-copy="code-{row['id']}">Copy code</button>
        <pre id="code-{row['id']}"><code>{esc(row['code'])}</code></pre>
        <p class="path">Verified source SHA-256: {esc(row['sha256'])}</p></details></article>''')
    for row in questions:
        search=esc((' '.join([row['id'],row['group'],row['question'],row['answer'],row['target']])).lower(),quote=True)
        cards.append(f'''<article class="card viva" data-kind="questions" data-search="{search}" id="{row['id']}">
        <div class="eyebrow">{esc(row['id'])} · {esc(row['group'])}</div><h3>{esc(row['question'])}</h3>
        <p>{esc(row['answer'])}</p><details><summary>Technical support &amp; claim limit</summary><p>{esc(row['technical'])}</p><p class="limit">{esc(row['limitation'])}</p></details>
        <a class="action" href="{esc(links[row['target']])}">Supporting block / {esc(row['target'])} ↗</a></article>''')
    topics=''.join(f'<a class="topic" href="{esc(n["html"])}"><b>{n["number"]}</b><span>{esc(n["title"])}</span><small>Executed · {n["cells"]} cells ↗</small></a>' for n in navigation['notebooks'])
    quick=''.join(f'<a href="{esc(links[i])}">{label} ↗</a>' for i,label in [('DEMO-RUN','Play / run demo'),('MODEL-ACCURACY','Model accuracy'),('BPDR-LOSSES','Loss functions'),('DATA-CLEAN','Data cleaning'),('MODEL-GATES','Failed validation'),('HANDOFF-EXTRACTOR','Extractor handoff')])
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>BPDR | Research notebook index</title><style>
    :root{--ink:#172c31;--muted:#506267;--paper:#f4f1e8;--teal:#206c67;--line:#d1d7cf;--accent:#b34f2a}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.5 'Segoe UI',sans-serif}
    header{background:var(--ink);color:#f6f3e8;padding:42px max(28px,calc((100vw - 1300px)/2));border-bottom:6px solid var(--teal)}
    .kicker,.eyebrow{font:12px/1.5 Consolas,monospace;text-transform:uppercase;letter-spacing:.08em}.kicker{color:#afd2c4}
    h1{font:normal clamp(38px,5vw,64px)/1.08 Georgia,serif;max-width:850px;margin:12px 0 18px}header p{max-width:850px;color:#d6dfd7;margin:0}
    main{max-width:1360px;margin:auto;padding:30px}.status{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:26px}.status span{font:13px Consolas,monospace;background:#e0e9df;padding:8px 12px;border-left:3px solid var(--teal)}
    .status .caution{background:#f1e4d6;border-color:var(--accent)}h2{font:normal 30px Georgia,serif;margin:34px 0 16px}.quick{display:flex;flex-wrap:wrap;gap:10px}.quick a{background:white;padding:10px 15px;border:1px solid var(--line);border-radius:3px}
    a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}a:focus-visible,button:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #d87733;outline-offset:3px}
    .topics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.topic{padding:18px;background:#ebece3;display:flex;flex-direction:column;border:1px solid var(--line)}.topic b{font:25px Georgia,serif;color:var(--accent)}.topic span{color:var(--ink);min-height:52px;margin:7px 0}.topic small{font-size:12px;color:var(--muted)}
    .searchbar{position:sticky;top:0;z-index:5;background:var(--paper);padding:16px 0 12px;border-bottom:1px solid var(--line)}label{display:block;font-weight:600;margin-bottom:8px}input{width:100%;padding:15px;background:white;border:1px solid var(--teal);border-radius:3px;color:var(--ink);font:18px Consolas,monospace}
    .filters{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}button{cursor:pointer;border:1px solid var(--line);background:white;color:var(--ink);padding:8px 13px;border-radius:3px;font:14px 'Segoe UI',sans-serif}button[aria-pressed=true]{background:var(--teal);border-color:var(--teal);color:white}#count{margin-left:auto;color:var(--muted);font-size:14px}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;padding-top:18px}.card{background:#fffefa;padding:23px;border:1px solid var(--line);border-top:3px solid var(--teal);scroll-margin-top:175px;min-width:0}.source{border-top-color:#7d8876}.viva{border-top-color:var(--accent)}.eyebrow{color:var(--muted)}h3{font:normal 23px/1.25 Georgia,serif;margin:9px 0 15px}.path{font:12px/1.6 Consolas,monospace;overflow-wrap:anywhere;color:var(--muted)}.action{font-weight:600;font-size:14px}details{margin:16px 0}summary{cursor:pointer;color:var(--teal)}pre{background:#14262b;color:#e4eee9;overflow:auto;max-height:500px;padding:18px;font:13px/1.6 Consolas,monospace;white-space:pre}.limit{border-left:3px solid var(--accent);padding-left:12px}.copy{margin-top:12px}article[hidden]{display:none}.empty{padding:25px;background:white}footer{padding:32px 0;color:var(--muted);font-size:14px}footer a{margin-right:18px}
    @media(max-width:950px){.topics{grid-template-columns:repeat(2,1fr)}}@media(max-width:640px){main{padding:18px}.grid{grid-template-columns:1fr}.topics{grid-template-columns:1fr}header{padding:28px 20px}.topic span{min-height:0}}
    @media print{header{color:black;background:white}.searchbar,.filters,.copy{display:none}.grid,.topics{display:block}.card{break-inside:avoid}article[hidden]{display:none}}
    </style><header><div class="kicker">J26-DS-339 / vocal input validation &amp; minimal repair</div><h1>Research notebook index.</h1><p>Find the explanation, code and evidence while exploring the implementation. Search a topic or a stable block ID; every result opens the exact executed walkthrough or the actual source function.</p></header><main>
    <div class="status"><span>''' + str(receipt['notebooks_completed']) + ''' notebooks executed</span><span>''' + str(receipt['errors']) + ''' execution errors</span><span>''' + str(receipt['plots']) + ''' plots · ''' + str(receipt['audio_players']) + ''' audio players</span><span class="caution">Experimental demo · research criteria still fail</span></div>
    <div class="quick">'''+quick+'''</div><h2>Choose a topic</h2><div class="topics">'''+topics+'''</div>
    <section aria-labelledby="find-title"><h2 id="find-title">Find code and evidence</h2><div class="searchbar"><label for="search">Search · press / to focus</label><input id="search" type="search" placeholder="Try: accuracy, loss, cleaning, rollback, DATA-FEATURES" autocomplete="off"><div class="filters" aria-label="Result type"><button type="button" data-filter="all" aria-pressed="true">All</button><button type="button" data-filter="blocks" aria-pressed="false">Notebook blocks</button><button type="button" data-filter="sources" aria-pressed="false">Source code</button><span id="count" aria-live="polite"></span></div></div><div class="grid">'''+''.join(cards)+'''</div><p id="empty" class="empty" hidden>No matching block. Try a shorter word, another topic or reset the result-type filter.</p></section>
    <footer><a href="../documentation/NOTEBOOK_GUIDE.md">Readiness report</a><a href="../documentation/EXPERIMENT_VERIFICATION_REPORT.md">Verification report</a><a href="../documentation/RESEARCH_STATUS.md">Research status</a><p>Prepared 20 October 2026 · local, offline reading views; no training or external service starts from this page. Source snippets are verified snapshots. To edit/run a notebook, use Run-Notebooks.ps1 and the BPDR local Python kernel. Human listening and independent natural-voice validation remain outstanding.</p><p id="copy-status" role="status"></p></footer></main><script>
    const search=document.getElementById('search'),cards=[...document.querySelectorAll('.card')],filters=[...document.querySelectorAll('[data-filter]')];let kind='all';
    function update(){const words=search.value.toLowerCase().trim().split(/\\s+/).filter(Boolean);let count=0;for(const card of cards){card.hidden=!((kind==='all'||card.dataset.kind===kind)&&words.every(w=>card.dataset.search.includes(w)));if(!card.hidden)count++;}document.getElementById('count').textContent=count+' results';document.getElementById('empty').hidden=count!==0;}
    search.addEventListener('input',update);filters.forEach(b=>b.addEventListener('click',()=>{kind=b.dataset.filter;filters.forEach(x=>x.setAttribute('aria-pressed',String(x===b)));update();}));
    document.addEventListener('keydown',event=>{if(event.key==='/'&&!['INPUT','TEXTAREA'].includes(document.activeElement.tagName)){event.preventDefault();search.focus();}if(event.key==='Escape'){search.value='';kind='all';filters.forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.filter==='all')));update();}});
    document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{const text=document.getElementById(button.dataset.copy).textContent;try{if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(text);}else{const field=document.createElement('textarea');field.value=text;document.body.appendChild(field);field.select();if(!document.execCommand('copy'))throw Error('Copy unavailable');field.remove();}document.getElementById('copy-status').textContent='Copied source code.';}catch(error){document.getElementById('copy-status').textContent='Copy unavailable in this browser. Select the code text and copy it manually.';}}));update();
    </script></html>'''
