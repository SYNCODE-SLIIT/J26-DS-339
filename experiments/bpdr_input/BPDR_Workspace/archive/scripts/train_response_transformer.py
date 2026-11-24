"""Matched response weighting, actual-WAV selection, then gated final refits."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import response_transformer_training as training
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.correction_accuracy import summarize, assert_independent
from bpdr_input.modeling.joint_transformer import load
import run_transformer_curve_study as curve

ROOT = WORKSPACE/'results/transformer_usability/2026-10-29_v12_response'
BASE = ROOT.parent/'2026-10-28_v8_joint'


def declare():
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        parent = ex.read(BASE/'declaration.json')
        files = [Path(__file__), Path(training.__file__),
                 WORKSPACE/'source/bpdr_input/modeling/joint_transformer.py',
                 WORKSPACE/'source/bpdr_input/experiments/learning/joint_transformer_training.py',
                 WORKSPACE/'source/bpdr_input/repair/curve_conditioning.py',
                 WORKSPACE/'source/bpdr_input/repair/framewise_audit.py',
                 WORKSPACE/'source/bpdr_input/repair/framewise_renderer.py', Path(curve.__file__)]
        ex.write(ROOT/'declaration.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
                 model_family='joint_context', arms=['matched_baseline', 'positive_weight_2'],
                 folds=parent['folds'], cv_seed=339, cv_updates=1200,
                 final_seeds=[339,340,341], default_seed=339, final_updates=2400,
                 loss='Only known nonzero-target squared anchor loss is weighted2 versus1; zero-target original weight4/outside2 and response/activity/amplitude losses unchanged. No temporal term.',
                 matching='Same initialization, deterministic sampler trace, architecture, 36 windows/update, optimization and1200 updates. Only positive target anchor weight differs:1 versus2. Exact V8 point-sampling scheme, no neighbor forwards.',
                 augmentation='Same 25% training-source feature-only pulse/drift/vibrato controls. Not new real WAVs or people.',
                 selection='First raw prediction gate; then all-original identity on actual OOF WAVs; then minimum singer-macro delivered error. Tie prefer matched baseline. No development-based selection.',
                 policy=curve.POLICY.__dict__, curve_condition='median11', renderer='Praat unchanged',
                 source_hashes={str(p):ex.sha(p) for p in files},
                 input_hashes=ex.read(BASE/'training_input_hashes.json'),
                 previous_final_data='IDs8/9 already inspected, excluded from fitting/selection and not fresh',
                 new_confirmation='male10/male11 and CSD female source en006a/en010a/en023a reserved; no confirmation audio accessed. Passing locked development and untouched confirmation still required before adoption.',
                 female_confirmation_reservation_sha256=ex.sha(ROOT.parent/'confirmation_sources_2026-10-29/csd_reservation.json'),
                 research_validation_passed=False, demo_activated=False))
        snapshot=ROOT/'executed_sources';snapshot.mkdir(exist_ok=True)
        for p in files:(snapshot/p.name).write_bytes(p.read_bytes())
        for name in ('train_pairs.json','anchor_sources.json'):(ROOT/name).write_bytes((BASE/name).read_bytes())
    d=ex.read(ROOT/'declaration.json')
    for name in ('source_hashes','input_hashes'):
        for p,h in d[name].items():assert ex.sha(p)==h,p
    return d


def delivery(d, arm):
    rows=ex.read(ROOT/'train_pairs.json');pairs=training.held.prepared(rows)
    cache,models={},{}
    for fold,groups in enumerate(d['folds'],1):
        folder=ROOT/f'cross_validation/fold_{fold}/{arm}'
        model,payload=load(folder/'checkpoint.pt')
        subset=[p for p in rows if ex.group(p) in groups]
        assert_independent(payload['training_groups'],payload['training_sources'],subset)
        for group in groups:models[group]=(model,payload)
        with np.load(folder/'validation/predictions.npz') as z:
            for i,p in enumerate(subset):
                if 'excluded' not in p:
                    for view in ('A','C'):cache[p['path_'+view]]=z[f'{i}_{view}'].copy()
    folder=ROOT/'arms'/arm;folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'declaration.json').exists():
        ex.write(folder/'declaration.json',dict(parent=str(ROOT/'declaration.json'),parent_sha256=ex.sha(ROOT/'declaration.json'),arm=arm))
    curve.ROOT=folder
    return curve.run_condition(rows,pairs,cache,models,ex.read(ROOT/'anchor_sources.json'),
                               'median11','training',ROOT/'train_pairs.json')


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['cv','delivery','refit'])
    args=parser.parse_args();d=declare()
    if args.stage=='cv':
        rows={arm:[] for arm in d['arms']}
        with ProcessPoolExecutor(max_workers=2) as pool:
            tasks={pool.submit(training.cv_worker,(str(ROOT),fold,arm)):(fold,arm)
                   for arm in d['arms'] for fold in range(len(d['folds']))}
            for future in as_completed(tasks):
                fold,arm=tasks[future];result=future.result();rows[arm].extend(result['rows'])
                print('RESPONSE_CV_COMPLETE',arm,fold+1,result['summary']['raw_mae_cents'],flush=True)
        results={arm:summarize(rr) for arm,rr in rows.items()}
        for fold in range(1,5):
            a=ex.read(ROOT/f'cross_validation/fold_{fold}/matched_baseline/fit_receipt.json')
            b=ex.read(ROOT/f'cross_validation/fold_{fold}/positive_weight_2/fit_receipt.json')
            assert a['sampling_sha256']==b['sampling_sha256'],'Mismatched training samples'
        ex.write(ROOT/'cross_validation.json',dict(results=results,sampling_matched_all_folds=True))
        print('RESPONSE_CV_FINISHED', {a:r['summary'] for a,r in results.items()},flush=True)
    elif args.stage=='delivery':
        cv=ex.read(ROOT/'cross_validation.json');results={}
        for arm in d['arms']:results[arm]=delivery(d,arm)
        selected=min(d['arms'],key=lambda a:(not cv['results'][a]['prediction_gate_passed'],
                     not results[a]['gates']['original_controls_all_identity'],
                     results[a]['summary']['delivered_after_mae_cents'],d['arms'].index(a)))
        ex.write(ROOT/'locked_selection.json',dict(arm=selected,results=results,
                 prediction_gate_passed=cv['results'][selected]['prediction_gate_passed'],
                 all_originals_identity=results[selected]['gates']['original_controls_all_identity'],
                 selection_data='Training-side raw gate and actual OOF WAV only',default_seed=339,
                 diagnostic_only=True,demo_activated=False,research_validation_passed=False))
        print('RESPONSE_DELIVERY_SELECTION',selected,flush=True)
    else:
        selection=ex.read(ROOT/'locked_selection.json')
        if not selection['prediction_gate_passed'] or not selection['all_originals_identity']:
            raise ValueError('Refits wait for prediction and original-preservation gates')
        pairs=training.held.prepared(ex.read(ROOT/'train_pairs.json'))
        anchors=training.held.anchor_pool(ex.read(ROOT/'anchor_sources.json'))
        for seed in d['final_seeds']:
            folder=ROOT/f'final/seed_{seed}'
            if (folder/'fit_receipt.json').exists():continue
            training.fit(pairs,anchors,selection['arm'],seed,d['final_updates'],folder,ex.sha(ROOT/'declaration.json'))
            print('RESPONSE_REFIT_COMPLETE',seed,flush=True)
    declare()


if __name__=='__main__':main()
