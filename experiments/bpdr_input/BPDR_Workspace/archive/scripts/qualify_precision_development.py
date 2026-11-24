"""Reproduce default full fit, then locked known-development precision-pipeline replay."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import evaluate_precision_auditor as parent
from bpdr_input.experiments.learning import joint_transformer_training as training
from bpdr_input.modeling.joint_transformer import load,predict_raw
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.experiments.learning.context_tree_training import sampled_queries

ex=parent.diagnostic.upstream.study.ex
V8=parent.curve.UPSTREAM
ROOT=parent.ROOT.parent/'2026-11-01_v34_precision_development'
DEV_BASE=parent.ROOT.parent/'2026-10-28_v9_curve/development/median11'


def declare():
    old=parent.declare();ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        assert ex.read(parent.ROOT/'integrity_receipt.json')['selected_training_side']
        inputs=dict(old['source_and_input_hashes'])
        for p in [V8/'declaration.json',V8/'training_source_hashes.json',V8/'training_input_hashes.json',parent.ROOT/'locked_selection.json',parent.ROOT/'integrity_receipt.json',DEV_BASE/'evidence.json',DEV_BASE/'accuracy.json',DEV_BASE/'predictions.npz',V8/'development/pairs.json']:
            inputs[str(p)]=ex.sha(p)
        for seed in (339,340,341):
            for name in ('checkpoint.pt','fit_receipt.json'):
                p=V8/f'final/seed_{seed}/{name}';inputs[str(p)]=ex.sha(p)
        sources=[Path(__file__).resolve(),Path(__file__).with_name('verify_precision_development.py').resolve()]
        for p in sources:inputs[str(p)]=ex.sha(p)
        ex.write(ROOT/'declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_and_input_hashes=inputs,parent_declaration_sha256=ex.sha(parent.ROOT/'declaration.json'),default_seed=339,full_updates=2400,
            lineage='Audit all three existing full checkpoint receipts/payloads against actual training-source/anchor pools, immutable source/input snapshots and CV config. Independently rerun original2400-update default339 fit from scratch using original source/data; require exact every-parameter reproduction before any development evaluation. Other seeds metadata/reload only and never selected from development. Original full fits did not save sampler digests; reconstruct the default schedule and label that fact explicitly.',
            development='Previously inspected IDs5/6 only; fixed default339,median11/Praat and V33 precision auditor.48 attempts,24 scoreable,24 excluded,12 original controls;two scoreable singer groups,two unassessed retained. No model/threshold/seed/policy/routing changes. Compare old frozen raw/request/permission arrays exactly, then actual encoded-WAV evidence under all original checks.',
            qualification='Require exact default reproduction, singer/source isolation,raw all-scoreable-singer gate,all12 originals identity,unchanged represented-singer practical gates and strictly lower singer-macro remaining delivered error than old18.04848509081958c. Independent verification before any fresh confirmation. No human/musical validation claim from development. No automatic activation.',
            fresh_confirmation_access=False,demo_activated=False,research_validation_passed=False))
    d=ex.read(ROOT/'declaration.json')
    for p,h in d['source_and_input_hashes'].items():assert ex.sha(p)==h,p
    for name in ('training_source_hashes','training_input_hashes'):
        for p,h in ex.read(V8/(name+'.json')).items():assert ex.sha(p)==h,p
    return d


def lineage(d):
    if (ROOT/'lineage_receipt.json').exists():raise ValueError('Retain completed full-fit lineage')
    rows=ex.read(V8/'train_pairs.json');pairs=training.held.prepared(rows);anchors=training.held.anchor_pool(ex.read(V8/'anchor_sources.json'))
    valid=[p for p in pairs if 'excluded' not in p]
    groups=sorted({p['group'] for p in valid+anchors});sources=sorted({p['source']['sha256'] for p in valid+anchors})
    development=ex.read(V8/'development/pairs.json');checks=[]
    for seed in (339,340,341):
        folder=V8/f'final/seed_{seed}';receipt=ex.read(folder/'fit_receipt.json');model,payload=load(folder/'checkpoint.pt')
        assert receipt['checkpoint_sha256']==ex.sha(folder/'checkpoint.pt') and receipt['reload_verified']
        for k,value in [('seed',seed),('updates',2400),('training_groups',groups),('training_sources',sources),('original_anchor_recordings',len(anchors)),('feature_only_training_examples',len(training.augmented_pairs(anchors,seed))),('declaration_sha256',ex.sha(V8/'declaration.json')),('arm','joint_augmented')]:
            assert receipt[k]==payload[k]==value,(seed,k)
        assert_independent(payload['training_groups'],payload['training_sources'],development)
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        for fold in range(1,5):
            _,cv=load(V8/f'cross_validation/fold_{fold}/joint_augmented/checkpoint.pt');assert cv['config']==payload['config'] and cv['feature_version']==payload['feature_version']
        checks.append(dict(seed=seed,checkpoint_sha256=ex.sha(folder/'checkpoint.pt'),training_groups=len(groups),training_sources=len(sources),independent_from_known_development=True))
    folder=ROOT/'reproduction/seed_339'
    if (folder/'fit_receipt.json').exists():
        rebuilt,rebuilt_payload=load(folder/'checkpoint.pt');assert ex.read(folder/'fit_receipt.json')['checkpoint_sha256']==ex.sha(folder/'checkpoint.pt')
    else:rebuilt,rebuilt_payload,_=training.fit(pairs,anchors,'joint_augmented',339,2400,folder,ex.sha(V8/'declaration.json'))
    original,payload=load(V8/'final/seed_339/checkpoint.pt')
    assert set(original.state_dict())==set(rebuilt.state_dict())
    assert all(torch.equal(v,rebuilt.state_dict()[k]) for k,v in original.state_dict().items()),'Default full fit differs: do not evaluate development'
    for key in ('training_groups','training_sources','seed','updates','feature_only_training_examples','original_anchor_recordings','config'):assert payload[key]==rebuilt_payload[key],key
    raw=valid[0]['A']['views'][0];np.testing.assert_array_equal(predict_raw(original,raw)[0],predict_raw(rebuilt,raw)[0])
    _,queries,_,_,digest,_=sampled_queries(pairs,anchors,339,2400)
    declare();ex.write(ROOT/'lineage_receipt.json',dict(checkpoints=checks,default_seed=339,default_state_tensors_exact=True,default_prediction_probe_exact=True,reconstructed_default_sampler_sha256=digest,reconstructed_default_draw_rows=len(queries),original_actual_draw_digest_available=False,original_inputs_source_hashes_verified=True,reproduced_full_fits=1,other_seeds_metadata_only=True,reproduction_checkpoint_sha256=ex.sha(folder/'checkpoint.pt'),declaration_sha256=ex.sha(ROOT/'declaration.json'),development_inference_started=False,demo_activated=False,research_validation_passed=False))
    print('DEFAULT_FULL_FIT_EXACT',len(queries),'rows',flush=True)


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
    print('PRECISION_DEVELOPMENT',qualified,'repairs',e['gates']['cohort_improved'],'after',e['summary']['delivered_after_mae_cents'],flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['declare','lineage','development']);args=parser.parse_args();d=declare()
    if args.stage=='lineage':lineage(d)
    elif args.stage=='development':development(d)
