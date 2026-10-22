"""Verify the six canonical notebooks in separate kernels; no training/downloads."""
import asyncio
import ast
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.reporting.research_support import read,sha,audit
from notebook_navigation import navigator

FORBIDDEN=re.compile(r'(?<![A-Za-z])(?:panel|presentations?)(?![A-Za-z])',re.I)


def reading_html(notebook):
    """Keep every result visible; make source code an optional disclosure."""
    page,_=HTMLExporter(exclude_input_prompt=True,exclude_output_prompt=True).from_notebook_node(notebook)
    soup=BeautifulSoup(page,'html.parser')
    for area in soup.select('.jp-CodeCell .jp-InputArea'):
        details=soup.new_tag('details',attrs={'class':'optional-code'})
        summary=soup.new_tag('summary')
        summary.string='Show code (optional)'
        area.wrap(details)
        details.insert(0,summary)
    style=soup.new_tag('style')
    style.string='''
    body{background:#fff;color:#172c31}
    main{max-width:1200px;margin:auto}
    details.optional-code{margin:8px 0 18px;border:1px solid #d5dfdd;border-radius:5px;padding:8px 12px}
    details.optional-code>summary{cursor:pointer;color:#24635f;font:14px system-ui}
    .jp-OutputArea-output{overflow-x:auto}
    .jp-RenderedHTMLCommon p,.jp-RenderedHTMLCommon li{line-height:1.65}
    .jp-RenderedHTMLCommon table td,.jp-RenderedHTMLCommon table th{color:#172c31!important}
    .jp-RenderedHTMLCommon table td{background:#fff!important}
    .jp-RenderedHTMLCommon table th{background:#e6efee!important}
    '''
    soup.head.append(style)
    nav=soup.new_tag('nav',style='padding:16px;background:#e7eee8;color:#172c31')
    link=soup.new_tag('a',href='index.html');link.string='← Notebook index';nav.append(link)
    nav.append(' · Results are visible. Expand code only when needed. Run All to reproduce with the BPDR local Python kernel.')
    soup.body.insert(0,nav)
    return str(soup)


def reports(receipt,checks):
    navigation=read(WORKSPACE/'notebooks/navigation.json')
    rows='\n'.join(f"| {n['number']} | [{n['title']}]({n['path']}) |" for n in navigation['notebooks'])
    (WORKSPACE/'notebooks/README.md').write_text(f'''# Research notebooks

Open [index.html](index.html) to search code blocks, view executed output or expand source functions. Run `../Open-Notebooks.ps1` to open the local index.

| Order | Notebook |
|---|---|
{rows}

Use the component `.venv` with the **BPDR local Python** kernel. Results, interpretation, plots and audio are visible on first open. Code is collapsed in HTML and marked collapsed in the notebook; expand it only when needed. Run all cells in a topic once before rerunning a dependent block. No fitting or dataset downloads run automatically.

Search IDs: `BPDR-LOSSES`, `DATA-CLEAN`, `DATA-FEATURES`, `MODEL-ACCURACY`, `MODEL-GATES`, `DEMO-RUN`, `RENDER-AUDIT`, `HANDOFF-EXTRACTOR`, `VERIFY-TESTS`.

Verified: {receipt['notebooks_completed']} notebooks, {receipt['code_cells']} code cells, {receipt['errors']} errors, {receipt['numerical_and_existing_tests']} passing tests, {receipt['plots']} plots and {receipt['audio_players']} audio players. Scientific acceptance failures remain visible.

Run `../.venv/Scripts/python.exe -X utf8 -B tools/execute_research_notebooks.py` from BPDR_Workspace to verify these six notebooks. Current execution receipt: `results/research_notebooks/current.json`. Superseded notebooks and exact user-edited originals were checksum-verified and archived under the location recorded in `results/notebook_cleanup/current.json`; they are no longer in the active notebook folder.

See [environment pins](../documentation/requirements-notebooks.txt), [usage guide](../documentation/NOTEBOOK_GUIDE.md), and [verification report](../documentation/EXPERIMENT_VERIFICATION_REPORT.md). Data, weights, historical receipts and generated outputs remain local and ignored by Git.
''',encoding='utf-8')
    (WORKSPACE/'documentation/NOTEBOOK_GUIDE.md').write_text('''# Notebook guide

Six independently executable topics cover foundations, methodology, data, models, audio processing and verification. Start with `notebooks/index.html` or `Open-Notebooks.ps1`. Search a code-block ID or function name; source cards show file paths, functions and line numbers.

Use saved HTML or the executed notebook to read existing outputs. Read each explanation, result and interpretation in order; expand optional code only when needed. The full implementation remains in source modules and optional index source cards, rather than being printed under result cells. To execute, open `Run-Notebooks.ps1`, select BPDR local Python and run the topic from its setup cells. `Open-Notebooks.ps1 -Verify` explicitly re-executes the six notebooks before opening the index. No training starts automatically.

Notebook 04 now includes explicit `MODEL-FIT-NEW`, `MODEL-FIT-EXECUTE`, `MODEL-SAVE-CHECKPOINT`, `MODEL-RELOAD-CHECKPOINT`, `MODEL-VERIFY-DEMO` and `MODEL-ACTIVATE-DEMO` steps. To deliberately fit a new fixed ExtraTrees reproduction, set `TRAIN_NEW_MODEL = True` in the first step, or launch verification with `BPDR_TRAIN_NOTEBOOK_MODEL=1`. Default execution reloads the previously fitted notebook checkpoint. Each fit writes a new directory under `results/model_recovery/notebook_training`; it never overwrites the historical checkpoint.

After train-only fitting, save/reload checks and a complete delivered-audio development replay, a separate `results/notebook_training/current.json` pointer selects the model used by both notebook 05 and the working demo app. Existing calibration/pilot data and locked settings are reused: this is reproducibility/development evidence, not an independent final test. The demo remains experimental and user review is required before export.

Model training remains deliberate: `python -B -m bpdr_input.experiments.learning.model_recovery` creates a new declared run. Keep earlier experiments and partitions unchanged. These are engineering development results; natural intended-note accuracy, blinded listening and downstream note-accuracy improvements remain unestablished.

The local-file browser preview could not be automated under the browser tool policy. Offline links and JavaScript search can be checked separately; test local browser/audio playback on the actual machine. This limitation does not change recorded model metrics.
''',encoding='utf-8')
    exceptions='\n'.join(f"| {c['check']} | {c['status']} | {c['actual']} |" for c in checks['checks'] if c['status']!='PASS')
    (WORKSPACE/'documentation/EXPERIMENT_VERIFICATION_REPORT.md').write_text(f'''# Verification report

{receipt['notebooks_completed']} independently executed notebooks; {receipt['code_cells']} code cells; {receipt['errors']} errors; {receipt['numerical_and_existing_tests']} passing automated tests. {receipt['plots']} plots and {receipt['audio_players']} audio players verified. Research acceptance remains incomplete.

Recomputed delivered-waveform metrics: {json.dumps(checks['pilot_summary'])}. These values retain every scoreable failure/withhold; exclusions and independent singer/recording counts remain explicit. No historical result, checkpoint, partition or acceptance threshold was changed by notebook cleanup.

| Check | Status | Observation |
|---|---|---|
{exceptions}

Archived notebook bytes and protected experiment hashes are checked separately from current notebook execution. Original recordings remain unchanged. Historical source locations are retained in the provenance manifests; superseded notebook sources can be restored from `results/notebook_cleanup/current.json`.
''',encoding='utf-8')
    (WORKSPACE/'documentation/RESEARCH_STATUS.md').write_text('''# Research status

The BPDR hypothesis separates robustness to boundary metadata from response to physical pitch changes. Four-view training, objectives and matched historical Transformer experiments are implemented. Reliable real-voice methodological superiority remains unestablished.

The optional ExtraTrees route is a separate engineering alternative: 96 trees and 31 own-audio context features. It was selected as a diagnostic calibration fallback after all declared full criteria failed. Excluding boundary metadata gives invariance by construction, not learned disentanglement. The model comparison uses common development examples with different checkpoint training histories.

Next work: independently reviewed intended-note/reference material, untouched singers and melodies, broader signed intervention coverage, matched neural training/evaluation, blinded listening and fixed-extractor note precision/recall/F1. Existing MIDI/CSV verifies interface execution, not extraction improvement. Exact source and model evidence remains unchanged.
''',encoding='utf-8')


def main():
    if sys.platform=='win32':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    navigation=read(WORKSPACE/'notebooks/navigation.json')
    history=read(WORKSPACE/'documentation/notebook_history.json')
    protected=history['protected_experiment_hashes']
    assert all(sha(p)==digest for p,digest in protected.items())
    assert sha(history['archived_original_notebook'])==history['archived_original_sha256']
    runtime=WORKSPACE/'results/notebook_runtime';kernel=runtime/'kernels/bpdr-local';kernel.mkdir(parents=True,exist_ok=True)
    temporary=runtime/'temporary';temporary.mkdir(exist_ok=True)
    env=dict(PYTHONPATH=str(WORKSPACE/'source'),PYTHONDONTWRITEBYTECODE='1',BPDR_COMPONENT_ROOT=str(WORKSPACE.parent),TEMP=str(temporary),TMP=str(temporary))
    os.environ.update(env)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-P','-B','-m','ipykernel_launcher','-f','{connection_file}'],display_name='BPDR local Python',language='python',env=env)),encoding='utf-8')
    for key,path in [('JUPYTER_PATH',runtime),('JUPYTER_RUNTIME_DIR',runtime/'runtime'),('JUPYTER_CONFIG_DIR',runtime/'config'),('IPYTHONDIR',runtime/'ipython')]:
        path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    out=WORKSPACE/'results/research_notebooks'/uuid.uuid4().hex;out.mkdir(parents=True)
    rendered={};rows=[];test_count=None;started=time.perf_counter()
    # Respect the user's current teaching edits. Never restore removed sections
    # merely to satisfy an old fixed output count.
    minima=[(1,3),(2,0),(3,0),(3,0),(4,5),(1,0)]
    for item,minimum in zip(navigation['notebooks'],minima):
        source=WORKSPACE/'notebooks'/item['path'];source_hash=sha(source)
        notebook=nbformat.read(source,4);nbformat.validate(notebook)
        # Old outputs may contain historical paths. Validate the current source,
        # then discard old outputs before fresh-kernel execution.
        assert not FORBIDDEN.search('\n'.join(c.source for c in notebook.cells)),source.name
        for c in notebook.cells:
            if c.cell_type=='code':
                ast.parse(c.source)
                # Always recompute from a clean kernel, including when the
                # reader-facing notebook already contains saved results.
                c.outputs=[];c.execution_count=None
        print('START',source.name,flush=True)
        client=NotebookClient(notebook,kernel_name='bpdr-local',timeout=180,
            timeout_func=lambda c:600 if '# Code block: MODEL-VERIFY-DEMO' in c.source else 180,
            startup_timeout=40,resources={'metadata':{'path':str(source.parent)}})
        client.on_cell_start=lambda cell,cell_index:print(cell.source.splitlines()[0],flush=True) if cell.cell_type=='code' else None
        try:client.execute()
        except Exception:
            nbformat.write(notebook,out/('FAILED_'+source.name));raise
        outputs=[o for c in notebook.cells for o in c.get('outputs',[])]
        errors=sum(o.get('output_type')=='error' for o in outputs)
        plots=sum('image/png' in o.get('data',{}) for o in outputs)
        audio=sum('<audio' in o.get('data',{}).get('text/html','') for o in outputs)
        if errors or plots<minimum[0] or audio<minimum[1]:raise RuntimeError('Missing output: '+source.name)
        # Inspect human-readable output, excluding binary image/audio encodings.
        for o in outputs:
            value=o.get('text','')+'\n'+o.get('data',{}).get('text/plain','')
            if FORBIDDEN.search(value):raise ValueError('Unwanted output wording in '+source.name+': '+FORBIDDEN.search(value).group())
            match=re.search(r'Executed tests:\s*(\d+)',value)
            if match:test_count=int(match.group(1))
        assert sha(source)==source_hash
        path=out/source.name;nbformat.write(notebook,path)
        page=reading_html(notebook)
        rendered[source.with_suffix('.html')]=page
        (out/source.with_suffix('.html').name).write_text(page,encoding='utf-8')
        # The canonical .ipynb should be useful on first open as well as the
        # HTML: retain freshly executed outputs and collapsed-input metadata.
        nbformat.write(notebook,source)
        rows.append(dict(notebook=str(source),source_sha256=sha(source),executed=str(path),executed_sha256=sha(path),
                         code_cells=sum(c.cell_type=='code' for c in notebook.cells),errors=errors,plots=plots,audio_players=audio))
        print('PASS',source.name,flush=True)
    for block in navigation['blocks']:
        file,anchor=block['html'].split('#');assert f'id="{anchor}"' in rendered[WORKSPACE/'notebooks'/file]
    assert all(sha(p)==digest for p,digest in protected.items())
    checks=audit();assert not any(c['scope']=='engineering' and c['status']=='FAIL' for c in checks['checks'])
    (out/'audit.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    receipt=dict(notebooks_completed=len(rows),notebooks=rows,code_cells=sum(r['code_cells'] for r in rows),errors=sum(r['errors'] for r in rows),
                 plots=sum(r['plots'] for r in rows),audio_players=sum(r['audio_players'] for r in rows),numerical_and_existing_tests=test_count,
                 seconds=round(time.perf_counter()-started,3),automatic_training=False,protected_experiments_unchanged=True,
                 archived_original_verified=True,research_validation_passed=False,
                 explicit_notebook_training=os.environ.get('BPDR_TRAIN_NOTEBOOK_MODEL')=='1',
                 active_notebook_demo=read(WORKSPACE/'results/notebook_training/current.json')
                    if (WORKSPACE/'results/notebook_training/current.json').exists() else None)
    index=navigator(navigation,receipt,[])
    rendered[WORKSPACE/'notebooks/index.html']=index
    for path,text in rendered.items():path.write_text(text,encoding='utf-8')
    (out/'index.html').write_text(index,encoding='utf-8')
    (out/'execution.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (out.parent/'current.json').write_text(json.dumps(dict(path=str(out),sha256=sha(out/'execution.json'),**receipt),indent=2),encoding='utf-8')
    reports(receipt,checks)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
