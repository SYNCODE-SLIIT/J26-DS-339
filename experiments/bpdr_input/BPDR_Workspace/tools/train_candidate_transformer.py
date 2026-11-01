"""Region-head fitting with frozen dense fallback; gated actual-WAV qualification."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np
import torch

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import candidate_head_training as training
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.candidate_transformer import load,predict_raw
from bpdr_input.modeling.correction_accuracy import summarize,assert_independent
import run_transformer_curve_study as curve

ROOT=WORKSPACE/'results/transformer_usability/2026-10-30_v22_candidate_head'
BASE=ROOT.parent/'2026-10-29_v12_response'
DATA=ROOT.parent/'2026-10-29_v21_candidate_data'


def declare():
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        reference=ex.read(BASE/'declaration.json');preflight=ex.read(DATA/'preflight.json')
        files=[Path(__file__).resolve(),Path(training.__file__),Path(curve.__file__),
               WORKSPACE/'source/bpdr_input/modeling/candidate_transformer.py',
               WORKSPACE/'source/bpdr_input/modeling/step_candidates.py',
               WORKSPACE/'source/bpdr_input/modeling/joint_transformer.py',
               WORKSPACE/'source/bpdr_input/modeling/held_voice_transformer.py',
               WORKSPACE/'source/bpdr_input/modeling/step_context.py',
               WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py',
               WORKSPACE/'source/bpdr_input/repair/curve_conditioning.py',
               WORKSPACE/'source/bpdr_input/repair/repair_controller.py',
               WORKSPACE/'source/bpdr_input/repair/framewise_audit.py',
               WORKSPACE/'source/bpdr_input/repair/framewise_renderer.py']
        data_hashes=dict(preflight['artifact_hashes'])
        for name in ('preflight.json','declaration.json'):data_hashes[str(DATA/name)]=ex.sha(DATA/name)
        cv_parents={str(f):str(BASE/f'cross_validation/fold_{f}/matched_baseline/checkpoint.pt') for f in range(1,5)}
        full_parents={str(s):str(ROOT.parent/f'2026-10-28_v8_joint/final/seed_{s}/checkpoint.pt') for s in (339,340,341)}
        paths=list(cv_parents.values())+list(full_parents.values())
        evidence=BASE/'arms/matched_baseline/training/median11/evidence.json'
        ex.write(ROOT/'declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
                 model_family='candidate_head_context',candidate_data=str(DATA),data_hashes=data_hashes,
                 architecture='Frozen V12/V8 Transformer dense parent; input-measured candidate tokens mean-pooled through frozen encoder and norm; two copied linear heads66 trainable parameters. Constant region override, unchanged dense predictions elsewhere. Not a new backbone fit.',
                 supervision='177 positive and8 real negative candidates, excluding14 unknown labels; paired original queries at the SAME input-derived candidate interval; four deterministic original-anchor queries per reliable source. All samples retain original singer/source folds.',
                 loss='Same normalized anchor/response/activity/amplitude terms as frame study;12 positive/original/anchor triples per update.25% extra-negative draws prefer real known negative candidates when available. No synthetic negative candidates fabricated.',
                 budget='Head-only1200 CV updates, AdamW0.001; conditional full-head refits2400 updates. Parent backbone remains byte-identical. Engineering candidate, not matched-budget objective/architecture superiority proof.',
                 folds=reference['folds'],cv_seed=339,cv_updates=1200,final_seeds=[339,340,341],default_seed=339,final_updates=2400,
                 cv_parents=cv_parents,full_parents=full_parents,parent_hashes={p:ex.sha(p) for p in paths},
                 policy=reference['policy'],curve_condition='median11',renderer='Praat unchanged',
                 selection='Raw all-singer prediction gate; all50 original WAVs identity; lower singer-macro delivered error than frozen reference. No change to numerical/output gates. Default339 fixed; no final/confirmation access.',
                 baseline_cv=str(BASE/'cross_validation.json'),baseline_cv_sha256=ex.sha(BASE/'cross_validation.json'),
                 baseline_evidence=str(evidence),baseline_evidence_sha256=ex.sha(evidence),
                 source_hashes={str(p):ex.sha(p) for p in files},input_hashes=ex.read(DATA/'declaration.json')['input_hashes'],
                 approved_training_sources=ex.read(ROOT.parent/'2026-10-29_v17_audio_pool/pool_declaration.json')['approved_sources'],
                 demo_activated=False,research_validation_passed=False))
        snapshot=ROOT/'executed_sources';snapshot.mkdir(exist_ok=True)
        for p in files:(snapshot/p.name).write_bytes(p.read_bytes())
        for name in ('train_pairs.json','anchor_sources.json'):(ROOT/name).write_bytes((BASE/name).read_bytes())
    d=ex.read(ROOT/'declaration.json')
    for label in ('source_hashes','input_hashes','parent_hashes','data_hashes'):
        for path,digest in d[label].items():assert ex.sha(path)==digest,path
    for label in ('baseline_cv','baseline_evidence'):assert ex.sha(d[label])==d[label+'_sha256']
    return d


def delivery(d):
    rows=ex.read(ROOT/'train_pairs.json');pairs=training.held.prepared(rows);cache={};models={}
    for fold,groups in enumerate(d['folds'],1):
        folder=ROOT/f'cross_validation/fold_{fold}/candidate_head';model,payload=load(folder/'checkpoint.pt')
        subset=[p for p in rows if ex.group(p) in groups]
        assert_independent(payload['training_groups'],payload['training_sources'],subset)
        for group in groups:models[group]=(model,payload)
        with np.load(folder/'validation/predictions.npz') as z:
            for i,p in enumerate(subset):
                if 'excluded' not in p:
                    for view in ('A','C'):cache[p['path_'+view]]=z[f'{i}_{view}'].copy()
    folder=ROOT/'arms/candidate_head';folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'declaration.json').exists():ex.write(folder/'declaration.json',dict(parent=str(ROOT/'declaration.json'),parent_sha256=ex.sha(ROOT/'declaration.json')))
    oldroot,oldpredict=curve.ROOT,curve.predict_raw;curve.ROOT=folder;curve.predict_raw=predict_raw
    try:return curve.run_condition(rows,pairs,cache,models,ex.read(ROOT/'anchor_sources.json'),'median11','training',ROOT/'train_pairs.json')
    finally:curve.ROOT,curve.predict_raw=oldroot,oldpredict


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['cv','delivery','refit']);args=parser.parse_args();d=declare()
    if args.stage=='cv':
        rows=[]
        with ProcessPoolExecutor(max_workers=2) as pool:
            tasks={pool.submit(training.cv_worker,(str(ROOT),f)):f for f in range(4)}
            for task in as_completed(tasks):
                fold=tasks[task];result=task.result();rows.extend(result['rows'])
                print('CANDIDATE_CV_COMPLETE',fold+1,result['summary']['raw_mae_cents'],flush=True)
        results=dict(matched_baseline=ex.read(d['baseline_cv'])['results']['matched_baseline'],candidate_head=summarize(rows))
        ex.write(ROOT/'cross_validation.json',dict(results=results,new_head_fits=4,new_backbone_fits=0))
        print('CANDIDATE_CV_FINISHED',results['candidate_head']['summary'],results['candidate_head']['prediction_gate_passed'],flush=True)
    elif args.stage=='delivery':
        evidence=delivery(d);cv=ex.read(ROOT/'cross_validation.json')['results']['candidate_head']
        baseline=ex.read(d['baseline_evidence'])
        selected=bool(cv['prediction_gate_passed'] and evidence['gates']['original_controls_all_identity']
                      and evidence['summary']['delivered_after_mae_cents']<baseline['summary']['delivered_after_mae_cents'])
        ex.write(ROOT/'locked_selection.json',dict(arm='candidate_head' if selected else 'matched_baseline',
                 prediction_gate_passed=cv['prediction_gate_passed'],all_originals_identity=evidence['gates']['original_controls_all_identity'],
                 default_seed=339,diagnostic_only=True,research_validation_passed=False,demo_activated=False))
        print('CANDIDATE_DELIVERY_SELECTION',selected,flush=True)
    else:
        choice=ex.read(ROOT/'locked_selection.json')
        if choice['arm']!='candidate_head' or not choice['prediction_gate_passed'] or not choice['all_originals_identity']:
            raise ValueError('No qualified new head selected; no refit')
        for seed in d['final_seeds']:
            folder=ROOT/f'final/seed_{seed}'
            if (folder/'fit_receipt.json').exists():continue
            training.fit(Path(d['candidate_data']),ex.read(ROOT/'anchor_sources.json'),set(),seed,d['final_updates'],folder,
                         ex.sha(ROOT/'declaration.json'),Path(d['full_parents'][str(seed)]))
            print('CANDIDATE_REFIT_COMPLETE',seed,flush=True)
    declare()


if __name__=='__main__':main()
