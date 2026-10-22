"""Keep six canonical topic notebooks and archive superseded local material.

All removals are individual, bounded files whose exact bytes have first been
copied and SHA-256 verified. Data, weights and experiment receipts are untouched.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid

import nbformat

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.reporting.panel_support import read,sha

TOPICS=[
 ('09','01_research_foundations.ipynb','Research foundations and music fundamentals'),
 ('10','02_bpdr_methodology.ipynb','BPDR methodology, Transformer and objectives'),
 ('11','03_data_preparation.ipynb','Data preparation, features and interventions'),
 ('12','04_model_training_and_evaluation.ipynb','Model training, comparison and evaluation'),
 ('13','05_audio_processing_and_handoff.ipynb','Audio processing, verification and handoff'),
 ('14','06_verification_and_statistics.ipynb','Evidence verification, statistics and tests')]
FORBIDDEN=re.compile(r'(?<![A-Za-z])(?:panel|presentations?)(?![A-Za-z])',re.I)


def neutral(text):
    text=text.replace('panel_support','research_support').replace('panel_source_manifest.json','research_sources.json')
    text=text.replace('execute_research_panel.py','execute_research_notebooks.py')
    text=text.replace('test_panel_support.py','test_research_support.py')
    text=re.sub(r'(?i)\bpanel desk\b','research index',text)
    text=re.sub(r'(?i)\bpanel\b','research',text)
    return text


def main():
    root=(WORKSPACE/'notebooks').resolve()
    archive=(WORKSPACE/'results/notebook_cleanup'/uuid.uuid4().hex).resolve()
    if not archive.is_relative_to((WORKSPACE/'results/notebook_cleanup').resolve()):raise ValueError('Archive leaves workspace')
    archive.mkdir(parents=True)
    # Preserve the user-edited source as well as the prior generated outputs.
    files=list(root.rglob('*'))
    files=[p for p in files if p.is_file()]
    old_tools=['build_research_panel.py','execute_research_panel.py','research_panel_content.py','check_panel_navigation.py','check_panel_search.js',
               'build_complete_recovery_notebook.py','execute_recovery_notebook.py','execute_usefulness_notebook.py','update_recovery_notebook.py']
    files += [WORKSPACE/'tools'/name for name in old_tools if (WORKSPACE/'tools'/name).is_file()]
    extra=[WORKSPACE/'Open-Research-Panel.ps1',WORKSPACE/'documentation/README.md']
    extra += [p for p in (WORKSPACE/'documentation').iterdir() if p.is_file() and ('panel' in p.name.lower() or p.name in ['EXPERIMENT_VERIFICATION_REPORT.md','RESEARCH_GAP_AND_IMPLEMENTATION_STATUS.md'])]
    files += [p for p in extra if p.is_file()]
    entries=[]
    for path in dict.fromkeys(files):
        resolved=path.resolve()
        if not resolved.is_relative_to(WORKSPACE.resolve()):raise ValueError('Snapshot source leaves workspace: '+str(path))
        backup=archive/'originals'/path.relative_to(WORKSPACE)
        backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(path.read_bytes())
        digest=sha(path)
        if sha(backup)!=digest:raise ValueError('Archive verification failed: '+str(path))
        entries.append(dict(original=str(path),archive=str(backup),sha256=digest,bytes=path.stat().st_size))
    preserved={r['original']:r for r in entries}
    old_navigation=read(root/'panel_navigation.json')
    model_run=Path(read(WORKSPACE/'results/model_recovery/current.json')['path'])
    protected={str(p):sha(p) for p in [model_run/'evidence.json',model_run/'selected_model.joblib',model_run/'locked_selection.json',
                 WORKSPACE/'results/model_recovery/demo_model.json',WORKSPACE/'selected_model/bpdr.pt']}
    original07=archive/'originals/notebooks/07_learned_stable_note_recovery.ipynb'
    config=dict(archive=str(archive),archived_original_notebook=str(original07),archived_original_sha256=sha(original07),
                protected_experiment_hashes=protected)
    (WORKSPACE/'documentation/notebook_history.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    # Neutral callable helper; the old helper stays intact for historical receipts.
    helper=(WORKSPACE/'source/bpdr_input/reporting/panel_support.py').read_text(encoding='utf-8')
    helper=neutral(helper)
    helper=helper.replace('WORKSPACE/"notebooks/07_learned_stable_note_recovery.ipynb"','archived_original()')
    helper=helper.replace("WORKSPACE/'notebooks/07_learned_stable_note_recovery.ipynb'",'archived_original()')
    helper+='''\n\ndef archived_original():
    history=read(WORKSPACE/'documentation/notebook_history.json')
    path=Path(history['archived_original_notebook'])
    if sha(path)!=history['archived_original_sha256']:
        raise ValueError('Archived original notebook changed')
    return path


def historical_assets():
    index=read(WORKSPACE/'notebooks/evidence_index.json')
    return dict(release_manifest=index['evidence']['presentation_manifest'],
                release_manifest_sha256=index['current_gate']['presentation_manifest_sha256'],
                adaptation_evidence=index['evidence']['presentation_adapt'])
'''
    (WORKSPACE/'source/bpdr_input/reporting/research_support.py').write_text(helper,encoding='utf-8')
    tests=(WORKSPACE/'tests/test_panel_support.py').read_text(encoding='utf-8')
    (WORKSPACE/'tests/test_research_support.py').write_text(neutral(tests).replace('PanelMetricTests','ResearchMetricTests'),encoding='utf-8')
    sources=read(WORKSPACE/'documentation/panel_source_manifest.json')
    (WORKSPACE/'documentation/research_sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    (WORKSPACE/'documentation/requirements-notebooks.txt').write_text(neutral((WORKSPACE/'documentation/requirements-panel.txt').read_text()),encoding='utf-8')
    notebooks=[];blocks=[];name_map={}
    for old_number,name,title in TOPICS:
        old=next(p for p in root.glob(old_number+'_*.ipynb'))
        name_map[old.name]=name
        n=nbformat.read(old,4)
        n.metadata.pop('panel',None)
        n.metadata['research']=dict(topic=name[:2],standalone=True,automatic_training=False)
        n.cells[0].source=f'# {name[:2]} · {title}\n\n[Notebook index](index.html)\n\nUse Restart Kernel and Run All once for this topic before rerunning a dependent block. Saved HTML includes code and outputs without a live kernel. No training or downloads run automatically.'
        for c in n.cells:
            c.source=neutral(c.source)
            if c.cell_type=='code':
                c.outputs=[];c.execution_count=None;c.metadata.pop('execution',None)
                c.source=c.source.replace('# research block:','# Code block:').replace('# Research block:','# Code block:')
                c.source=c.source.replace("historical_index['evidence']['presentation_manifest']","HISTORY_PATHS['release_manifest']")
                c.source=c.source.replace("historical_index['current_gate']['presentation_manifest_sha256']","HISTORY_PATHS['release_manifest_sha256']")
                c.source=c.source.replace("historical_index['evidence']['presentation_adapt']","HISTORY_PATHS['adaptation_evidence']")
                c.source=c.source.replace("if k!='frozen'","if k not in ('frozen','version')")
                if "from bpdr_input.reporting.research_support import fixed_metrics,paired_reduction" in c.source:
                    c.source=c.source.replace('import fixed_metrics,paired_reduction','import fixed_metrics,paired_reduction,historical_assets')
                    c.source+='\nHISTORY_PATHS=historical_assets()'
                if "patterns=['test_research_support.py'" in c.source:
                    c.source=c.source.replace("'test_research_support.py'","'test_research_support.py'")
                ast.parse(c.source)
            meta=c.metadata.pop('panel_block',None)
            if meta:
                meta['title']=neutral(meta['title']);c.metadata['code_block']=meta
                blocks.append(dict(id=meta['id'],title=meta['title'],notebook=name,cell=len([None]),html=name.replace('.ipynb','.html')+'#'+meta['id']))
            if FORBIDDEN.search(c.source):raise ValueError('Unwanted wording remains in '+name+': '+FORBIDDEN.search(c.source).group())
        for i,c in enumerate(n.cells):
            meta=c.metadata.get('code_block')
            if meta:next(b for b in reversed(blocks) if b['id']==meta['id'] and b['notebook']==name)['cell']=i
        n.metadata.pop('widgets',None);nbformat.validate(n);nbformat.write(n,root/name)
        notebooks.append(dict(number=name[:2],title=title,path=name,html=name.replace('.ipynb','.html'),cells=len(n.cells)))
    code_cards=old_navigation['sources']
    for card in code_cards:
        if card['path'].endswith('/panel_support.py'):
            card['path']=card['path'].replace('panel_support','research_support')
            text=(WORKSPACE/card['path']).read_text();tree=ast.parse(text)
            node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==card['function'])
            card.update(line=node.lineno,end_line=node.end_lineno,sha256=sha(WORKSPACE/card['path']),code=ast.get_source_segment(text,node))
        card['title']=neutral(card['title'])
    navigation=dict(version='research-notebooks-1',notebooks=notebooks,blocks=blocks,sources=code_cards,
                    text_anchors=['LITERATURE','HANDOFF-CONTRACT'],history='documentation/notebook_history.json')
    (root/'navigation.json').write_text(json.dumps(navigation,indent=2),encoding='utf-8')
    index=read(root/'evidence_index.json');index['maintained_notebooks']=[r['path'] for r in notebooks]
    index['purpose']='Six canonical executable research topics with saved-model evaluation and audio handoff verification.'
    (root/'evidence_index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    # Reuse the tested navigator, with neutral labels and only relevant cards.
    executor=(WORKSPACE/'tools/execute_research_panel.py').read_text(encoding='utf-8')
    node=next(n for n in ast.parse(executor).body if isinstance(n,ast.FunctionDef) and n.name=='navigator')
    nav_source=ast.get_source_segment(executor,node)
    nav_source=nav_source.replace('15_Panel_Questions_Rehearsal_and_Roadmap.html','01_research_foundations.html')
    nav_source=nav_source.replace('13_Live_Audio_Controller_and_Handoff.html','05_audio_processing_and_handoff.html')
    nav_source=nav_source.replace('09_Research_Music_and_Literature.html','01_research_foundations.html')
    nav_source=nav_source.replace('Your research panel desk.','Research notebook index.').replace('while you present','while exploring the implementation')
    nav_source=nav_source.replace('PANEL_READINESS_REPORT.md','NOTEBOOK_GUIDE.md')
    nav_source=nav_source.replace('RESEARCH_GAP_AND_IMPLEMENTATION_STATUS.md','RESEARCH_STATUS.md')
    nav_source=nav_source.replace('<button type="button" data-filter="questions" aria-pressed="false">Panel questions</button>','')
    nav_source=nav_source.replace('rehearsing the actual machine','checking the local environment')
    nav_source=nav_source.replace('presentation machine','local machine').replace('presentation','research')
    nav_source=neutral(nav_source)
    nav_source=nav_source.replace('prepared 20 October 2026','verified research notebooks')
    (WORKSPACE/'tools/notebook_navigation.py').write_text('import html\n\n'+nav_source+'\n',encoding='utf-8')
    # All superseded active notebook files are removed only after backup checks.
    to_remove=[p for p in files if p.is_relative_to(root) and p.name not in ('README.md','evidence_index.json')]
    to_remove += [WORKSPACE/'tools'/name for name in old_tools if (WORKSPACE/'tools'/name).exists()]
    to_remove += [p for p in extra if p.exists() and ('panel' in p.name.lower() or p.name=='Open-Research-Panel.ps1')]
    removed=[]
    for path in dict.fromkeys(to_remove):
        if str(path) not in preserved:raise ValueError('Removal lacks verified backup: '+str(path))
        row=preserved[str(path)]
        if not path.resolve().is_relative_to(WORKSPACE.resolve()) or sha(path)!=row['sha256'] or sha(row['archive'])!=row['sha256']:
            raise ValueError('Unsafe or changed removal target: '+str(path))
        path.unlink();removed.append(str(path))
    if not all(sha(p)==h for p,h in protected.items()):raise ValueError('Protected experiment changed')
    receipt=dict(archive=str(archive),snapshots=entries,removed_from_active_workspace=removed,
                 retained_notebooks=[n['path'] for n in notebooks],protected_experiment_hashes=protected,
                 scope='Active notebook cleanup only; source data, checkpoints and old experiment receipts preserved')
    (archive/'cleanup.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (WORKSPACE/'results/notebook_cleanup/current.json').write_text(json.dumps(dict(path=str(archive),receipt_sha256=sha(archive/'cleanup.json')),indent=2),encoding='utf-8')
    print(json.dumps(dict(notebooks=len(notebooks),blocks=len(blocks),removed_files=len(removed),archive=str(archive)),indent=2))


if __name__=='__main__':main()
