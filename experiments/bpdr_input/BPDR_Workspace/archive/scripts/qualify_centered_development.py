"""Locked known-development qualification of the training-selected centered auditor."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import evaluate_centered_auditor as parent
import qualify_precision_development as prior
from bpdr_input.modeling.correction_accuracy import assert_independent

ex=prior.ex
V8=prior.V8
ROOT=prior.ROOT.parent/'2026-11-01_v36_centered_development'
DEV_BASE=prior.ROOT/'arm/development/median11'
training=prior.training
load=prior.load
predict_raw=prior.predict_raw


def declare():
    selected=parent.declare();prior.declare();ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        assert ex.read(parent.ROOT/'integrity_receipt.json')['selected_training_side']
        lineage=ex.read(prior.ROOT/'lineage_integrity_receipt.json');assert lineage['default_state_tensors_exact']
        for name in ('lineage_receipt.json','lineage_integrity_receipt.json'):(ROOT/name).write_bytes((prior.ROOT/name).read_bytes())
        inputs=dict(selected['source_and_input_hashes'])
        paths=[Path(__file__).resolve(),Path(__file__).with_name('verify_centered_development.py').resolve(),parent.ROOT/'integrity_receipt.json',parent.ROOT/'locked_selection.json',prior.ROOT/'integrity_receipt.json',prior.ROOT/'qualification_status.json',prior.ROOT/'lineage_receipt.json',prior.ROOT/'lineage_integrity_receipt.json',V8/'final/seed_339/checkpoint.pt',V8/'development/pairs.json',DEV_BASE/'accuracy.json',DEV_BASE/'predictions.npz',DEV_BASE/'evidence.json']
        for p in paths:inputs[str(p)]=ex.sha(p)
        inputs.update(ex.input_hashes(ex.read(V8/'development/pairs.json')))
        for source_file in sorted((V8.parent/'2026-10-28_v7_guard/development').glob('source_*.json')):
            source=ex.read(source_file)['source'];path=Path(source['working_audio'])
            for p in (source_file,path,path.parent/'original_features.npz'):inputs[str(p)]=ex.sha(p)
        ex.write(ROOT/'declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_and_input_hashes=inputs,parent_training_selection_sha256=ex.sha(parent.ROOT/'locked_selection.json'),default_seed=339,full_fit_lineage_reused_from=str(prior.ROOT),new_model_fits=0,
            protocol='Training-selected centered ALWA arithmetic only, same frozen default339 model. Lock before known-development replay. All48 attempts/24 scoreable/24 excluded,12 originals,raw predictions,median11/Praat,permission decisions,supports and every existing numerical/sign limit remain unchanged. No seed/model/decoder routing or threshold changes. Prior development has been inspected; this is not fresh confirmation.',
            qualification='Require reused exact full-fit default reproduction and source/singer isolation,raw all-scoreable-singer gate,all12 originals identity,unchanged practical represented-singer gates and strictly lower singer-macro delivered remaining error than prior V34 precision pipeline17.322854125070776c. Independently replay model predictions,all arrays/decisions,candidate/encoded audits and24 reference scores/gates. Female6/male5 stay unassessed. A scoped pass is not broad usability and cannot substitute for the unchanged fresh4-voice2F/2M cohort or human listening.',
            fresh_confirmation_access=False,demo_activated=False,research_validation_passed=False))
    d=ex.read(ROOT/'declaration.json')
    for p,h in d['source_and_input_hashes'].items():assert ex.sha(p)==h,p
    for name in ('lineage_receipt.json','lineage_integrity_receipt.json'):assert ex.sha(ROOT/name)==ex.sha(prior.ROOT/name)
    assert d['parent_training_selection_sha256']==ex.sha(parent.ROOT/'locked_selection.json')
    return d


def development(d):
    receipt=ex.read(ROOT/'lineage_receipt.json');assert receipt['default_state_tensors_exact']
    independent=ex.read(ROOT/'lineage_integrity_receipt.json');assert independent['default_state_tensors_exact'] and independent['lineage_receipt_sha256']==ex.sha(ROOT/'lineage_receipt.json')
    if (ROOT/'development_decision.json').exists():raise ValueError('Retain completed known-development replay')
    lock=ROOT/'development_lock.json'
    if not lock.exists():ex.write(lock,dict(created_at=datetime.now(timezone.utc).isoformat(),declaration_sha256=ex.sha(ROOT/'declaration.json'),lineage_receipt_sha256=ex.sha(ROOT/'lineage_receipt.json'),checkpoint_sha256=ex.sha(V8/'final/seed_339/checkpoint.pt'),seed=339,policy=parent.curve.POLICY.__dict__,condition='median11',known_development_only=True,fresh_confirmation_access=False))
    manifest=V8/'development/pairs.json';rows=ex.read(manifest);model,payload=load(V8/'final/seed_339/checkpoint.pt');assert_independent(payload['training_groups'],payload['training_sources'],rows)
    pairs=training.held.prepared(rows);cache={}
    for p in pairs:
        if 'excluded' in p:continue
        for view in ('A','C'):
            if p['path_'+view] not in cache:cache[p['path_'+view]]=np.stack([predict_raw(model,p[view]['views'][0])[0]]*3)
    with np.load(DEV_BASE/'predictions.npz') as z:
        for i,p in enumerate(pairs):
            if 'excluded' not in p:
                for view in ('A','C'):np.testing.assert_array_equal(cache[p['path_'+view]],z[f'{i}_{view}'])
    sources=[ex.read(p)['source'] for p in sorted((V8.parent/'2026-10-28_v7_guard/development').glob('source_*.json'))]
    assert len(rows)==48 and sum('excluded' not in r for r in rows)==24 and len(sources)==12
    models={s['source']['singer_id']:(model,payload) for s in sources}
    folder=ROOT/'arm';folder.mkdir(exist_ok=True)
    if not (folder/'declaration.json').exists():ex.write(folder/'declaration.json',dict(parent=str(ROOT/'declaration.json'),parent_sha256=ex.sha(ROOT/'declaration.json')))
    oldroot=parent.curve.ROOT;oldrender=ex.rendered_output;measure_cache={}
    ex.rendered_output=lambda p,v,c,plan,out:parent.precision_audit.rendered_output(p,v,c,plan,out,ex.COMPONENT_ROOT/'checkpoints',measure_cache)
    parent.curve.ROOT=folder
    try:e=parent.curve.run_condition(rows,pairs,cache,models,sources,'median11','development',manifest)
    finally:parent.curve.ROOT=oldroot;ex.rendered_output=oldrender
    raw=ex.read(folder/'development/median11/accuracy.json')['prediction_gate_passed'];old=ex.read(DEV_BASE/'evidence.json')
    qualified=bool(raw and e['gates']['passed'] and e['gates']['original_controls_all_identity'] and e['summary']['delivered_after_mae_cents']<old['summary']['delivered_after_mae_cents'])
    declare();ex.write(ROOT/'development_decision.json',dict(qualified_on_known_scoreable_scope=qualified,raw_gate_passed=raw,practical_gates_passed=e['gates']['passed'],default_seed=339,originals_identity=e['gates']['original_controls_all_identity'],fresh_confirmation_access=False,independent_verification_pending=True,demo_activated=False,research_validation_passed=False))
    print('CENTERED_DEVELOPMENT',qualified,'repairs',e['gates']['cohort_improved'],'after',e['summary']['delivered_after_mae_cents'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);d=declare();development(d)
