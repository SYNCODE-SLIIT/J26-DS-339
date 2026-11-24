"""Compare multiscale context with uniform or difficult-original anchors."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import multiscale_transformer_training as training
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.multiscale_transformer import load,predict_raw
from bpdr_input.modeling.correction_accuracy import summarize,assert_independent
import run_transformer_curve_study as curve

ROOT=WORKSPACE/'results/transformer_usability/2026-10-29_v14_multiscale'
BASE=ROOT.parent/'2026-10-29_v12_response'
ARMS=['multiscale_uniform','multiscale_hard']


def declare():
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        parent=ex.read(BASE/'declaration.json')
        evidence=BASE/'arms/matched_baseline/training/median11/evidence.json'
        files=[Path(__file__),Path(training.__file__),Path(curve.__file__),
               WORKSPACE/'source/bpdr_input/modeling/multiscale_transformer.py',
               WORKSPACE/'source/bpdr_input/modeling/held_voice_transformer.py',
               WORKSPACE/'source/bpdr_input/modeling/joint_transformer.py',
               WORKSPACE/'source/bpdr_input/modeling/step_context.py',
               WORKSPACE/'source/bpdr_input/modeling/step_transformer.py',
               WORKSPACE/'source/bpdr_input/experiments/learning/joint_transformer_training.py',
               WORKSPACE/'source/bpdr_input/experiments/learning/response_transformer_training.py',
               WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py',
               WORKSPACE/'source/bpdr_input/repair/curve_conditioning.py',
               WORKSPACE/'source/bpdr_input/repair/framewise_audit.py',
               WORKSPACE/'source/bpdr_input/repair/framewise_renderer.py']
        d=dict(created_at=datetime.now(timezone.utc).isoformat(),model_family='multiscale_context',
               arms=['matched_baseline']+ARMS,trained_arms=ARMS,folds=parent['folds'],cv_seed=339,cv_updates=1200,
               final_seeds=[339,340,341],default_seed=339,final_updates=2400,
               model='13 input channels:5 broad context,5 current acoustic-region context,3 step cues. Same two-layer width32/heads4 FF96,51 tokens.160 more projection parameters than the8-channel reference. Broad attention remains available.',
               original_sampling='Uniform arm uses original sampler. Hard arm replaces50% of original-anchor frame draws using reliable local-residual80th-percentile-or-higher frames from the SAME chosen training original. Separate RNG; no target/reference/model score enters mining.',
               matching='Two new arms have identical initialization, architecture, paired samples, source choices, loss weights, optimizer and1200 updates. Uniform-reference sampler trace must match V12 in every fold; hard draws are separately recorded.',
               selection=parent['selection'],policy=parent['policy'],curve_condition='median11',renderer='Praat unchanged',
               baseline_evidence=str(evidence),baseline_evidence_sha256=ex.sha(evidence),
               baseline_cv=str(BASE/'cross_validation.json'),baseline_cv_sha256=ex.sha(BASE/'cross_validation.json'),
               baseline_checkpoints={str(BASE/f'cross_validation/fold_{f}/matched_baseline/checkpoint.pt'):
                   ex.sha(BASE/f'cross_validation/fold_{f}/matched_baseline/checkpoint.pt') for f in range(1,5)},
               source_hashes={str(p):ex.sha(p) for p in files},input_hashes=parent['input_hashes'],
               new_confirmation=parent['new_confirmation'],research_validation_passed=False,demo_activated=False)
        ex.write(ROOT/'declaration.json',d)
        snapshot=ROOT/'executed_sources';snapshot.mkdir(exist_ok=True)
        for p in files:(snapshot/p.name).write_bytes(p.read_bytes())
        for name in ('train_pairs.json','anchor_sources.json'):(ROOT/name).write_bytes((BASE/name).read_bytes())
    d=ex.read(ROOT/'declaration.json')
    for name in ('source_hashes','input_hashes','baseline_checkpoints'):
        for p,h in d[name].items():assert ex.sha(p)==h,p
    assert ex.sha(d['baseline_evidence'])==d['baseline_evidence_sha256']
    assert ex.sha(d['baseline_cv'])==d['baseline_cv_sha256']
    return d


def delivery(d,arm):
    rows=ex.read(ROOT/'train_pairs.json');pairs=training.held.prepared(rows);cache={};models={}
    for fold,groups in enumerate(d['folds'],1):
        folder=ROOT/f'cross_validation/fold_{fold}/{arm}';model,payload=load(folder/'checkpoint.pt')
        subset=[p for p in rows if ex.group(p) in groups]
        assert_independent(payload['training_groups'],payload['training_sources'],subset)
        for group in groups:models[group]=(model,payload)
        with np.load(folder/'validation/predictions.npz') as z:
            for i,p in enumerate(subset):
                if 'excluded' not in p:
                    for view in ('A','C'):cache[p['path_'+view]]=z[f'{i}_{view}'].copy()
    folder=ROOT/'arms'/arm;folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'declaration.json').exists():ex.write(folder/'declaration.json',dict(parent=str(ROOT/'declaration.json'),parent_sha256=ex.sha(ROOT/'declaration.json'),arm=arm))
    old_root,old_predict=curve.ROOT,curve.predict_raw
    curve.ROOT=folder;curve.predict_raw=predict_raw
    try:return curve.run_condition(rows,pairs,cache,models,ex.read(ROOT/'anchor_sources.json'),'median11','training',ROOT/'train_pairs.json')
    finally:curve.ROOT,curve.predict_raw=old_root,old_predict


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['cv','delivery','refit'])
    args=parser.parse_args();d=declare()
    if args.stage=='cv':
        rows={a:[] for a in ARMS}
        with ProcessPoolExecutor(max_workers=2) as pool:
            tasks={pool.submit(training.cv_worker,(str(ROOT),f,a)):(f,a) for a in ARMS for f in range(4)}
            for task in as_completed(tasks):
                fold,arm=tasks[task];result=task.result();rows[arm].extend(result['rows'])
                reference=ex.read(BASE/f'cross_validation/fold_{fold+1}/matched_baseline/fit_receipt.json')
                receipt=ex.read(ROOT/f'cross_validation/fold_{fold+1}/{arm}/fit_receipt.json')
                assert receipt['uniform_reference_sampling_sha256']==reference['sampling_sha256']
                print('MULTISCALE_CV_COMPLETE',arm,fold+1,result['summary']['raw_mae_cents'],flush=True)
        for f in range(1,5):
            a=ex.read(ROOT/f'cross_validation/fold_{f}/{ARMS[0]}/fit_receipt.json')
            b=ex.read(ROOT/f'cross_validation/fold_{f}/{ARMS[1]}/fit_receipt.json')
            assert a['paired_sampling_sha256']==b['paired_sampling_sha256']
        results={'matched_baseline':ex.read(d['baseline_cv'])['results']['matched_baseline']}
        results.update({a:summarize(rr) for a,rr in rows.items()})
        ex.write(ROOT/'cross_validation.json',dict(results=results,new_neural_fits=8,reused_reference_fits=4,
                 paired_samples_match=True,reference_sampling_matches=True))
        print('MULTISCALE_CV_FINISHED',{a:v['summary'] for a,v in results.items()},flush=True)
    elif args.stage=='delivery':
        cv=ex.read(ROOT/'cross_validation.json');results={'matched_baseline':ex.read(d['baseline_evidence'])}
        for arm in ARMS:results[arm]=delivery(d,arm)
        selected=min(d['arms'],key=lambda a:(not cv['results'][a]['prediction_gate_passed'],
                     not results[a]['gates']['original_controls_all_identity'],
                     results[a]['summary']['delivered_after_mae_cents'],d['arms'].index(a)))
        ex.write(ROOT/'locked_selection.json',dict(arm=selected,results=results,
                 prediction_gate_passed=cv['results'][selected]['prediction_gate_passed'],
                 all_originals_identity=results[selected]['gates']['original_controls_all_identity'],
                 default_seed=339,diagnostic_only=True,selection_data='Training-side prediction/original gates then actual WAV error',
                 demo_activated=False,research_validation_passed=False))
        print('MULTISCALE_DELIVERY_SELECTION',selected,flush=True)
    else:
        selected=ex.read(ROOT/'locked_selection.json')
        if selected['arm'] not in ARMS:raise ValueError('No new condition selected; no duplicate baseline refit')
        if not selected['prediction_gate_passed'] or not selected['all_originals_identity']:raise ValueError('Prediction/original gates failed; no refit')
        pairs=training.held.prepared(ex.read(ROOT/'train_pairs.json'));anchors=training.held.anchor_pool(ex.read(ROOT/'anchor_sources.json'))
        for seed in d['final_seeds']:
            folder=ROOT/f'final/seed_{seed}'
            if (folder/'fit_receipt.json').exists():continue
            training.fit(pairs,anchors,selected['arm'],seed,d['final_updates'],folder,ex.sha(ROOT/'declaration.json'))
            print('MULTISCALE_REFIT_COMPLETE',seed,flush=True)
    declare()


if __name__=='__main__':main()
