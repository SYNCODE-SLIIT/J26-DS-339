"""Prepare newly declared TRAINING sources only; reserved singers stay unread."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.experiments.learning.context_improvement import protected_files,sha

ROOT=WORKSPACE/'results/transformer_usability/2026-10-28_v1'


def worker(task):
    import torch
    torch.set_num_threads(2)
    index,row=task
    receipt=ROOT/f'source_{index:02d}.json'
    if receipt.exists(): return json.loads(receipt.read_text())
    source,pairs=recovery.prepare_source(row,ROOT,index)
    record=dict(source=source,pairs=[recovery.serialise_pair(p) for p in pairs])
    recovery.write(receipt,record)
    return record


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    dataset=WORKSPACE/'datasets/vocalset_expansion_2026-10-28'
    manifest=recovery.read(dataset/'manifest.json')
    assert manifest['status']=='COMPLETE'
    rows=[r for r in manifest['records'] if r['split']=='train']
    declaration=ROOT/'declaration.json'
    if not declaration.exists():
        recovery.write(declaration,dict(created_at=datetime.now(timezone.utc).isoformat(),
            version='held-voice-usability-1',source_reservation_sha256=sha(dataset/'declaration.json'),
            training_sources=rows,protected_files=protected_files(),
            final_singers=['female9','male9'],development_singers=['female8','male8'],
            final_access='No target creation or inference before locked development gates pass',
            model='Input-only 4-second, stride-4, 51-token, 2-layer width-32 Transformer; signed bounded amplitude and separately learned activity; no chromatic pitch classes',
            training='Two declared arms gated/regression, seed339, four singer folds, 800 updates; select by original<=3 per singer then raw error, otherwise diagnostic weighted metric; refit chosen arm seeds339/340/341 1200 updates, fixed default339',
            folds=[['female1','male1','female7'],['female2','male2','male7'],['female3','male3'],['female4','male4']],
            loss='Normalized squared correction; unchanged-original weight4; known outside weight2; supervised activity BCE .25; paired response .25; same-audio boundary consistency .1; center sign/detuning labels only on audited mask',
            policy_grid=dict(activation_cents=[8.,12.,16.],release_fraction=.5,gap_frames=[0,2,4],min_support_seconds=.86,max_correction_cents=30.,max_changed_fraction=.25,max_disagreement_cents=5.),
            policy='All sign-consistent candidate spans, no padding; unreliable/energy onset/large pitch transitions/protected audio veto; acoustic held-voice eligibility separate from provisional pitch boundaries; policy selection uses training only',
            practical_gates='Per represented independent singer: original raw request mean<=3c; delivered original identity; >=25% all-attempt verified improved interventions; both shift directions represented overall; verified delivered gain>=25%; fixed numerical/audio audits unchanged',
            research_validation_passed=False,demo_activated=False))
    old=Path(recovery.read(WORKSPACE/'results/model_recovery/current.json')['path'])
    pairs=recovery.read(old/'train_pairs.json')
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i,record in enumerate(pool.map(worker,enumerate(rows,1)),1):
            pairs.extend(record['pairs'])
            recovery.write(ROOT/'preparation_progress.json',dict(completed_sources=i,planned_sources=len(rows),attempted_interventions=len(pairs),scoreable_interventions=sum('excluded' not in p for p in pairs)))
            print('PREPARED',i,'/',len(rows),'scoreable',sum('excluded' not in p for p in pairs),'/',len(pairs),flush=True)
    recovery.write(ROOT/'train_pairs.json',pairs)
    print('TRAINING_PREPARATION_COMPLETE',len(pairs),flush=True)
