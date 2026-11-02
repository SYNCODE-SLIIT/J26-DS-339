"""Matched nearby known-zero sampling with frozen output checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import torch
WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import nearby_zero_training as training
from bpdr_input.modeling.correction_accuracy import summarize, assert_independent
import train_absolute_response_transformer as control
import run_transformer_curve_study as curve
ex=control.ex
BASE=control.BASE
ROOT=control.ROOT.parent/'2026-11-01_v32_nearby_zero'


def declare():
    ROOT.mkdir(exist_ok=True)
    parent=control.declare()
    if not (ROOT/'declaration.json').exists():
        assert ex.read(control.ROOT/'integrity_receipt.json')['controls_reproduced_exact']==4
        for fold in range(1,5):
            control.verify_control(fold)
        retained=('folds','default_seed','policy','curve_condition','renderer','input_hashes','baseline_cv','baseline_cv_sha256','baseline_evidence','baseline_evidence_sha256')
        d={k:parent[k] for k in retained}
        sources=dict(parent['source_hashes'])
        for path in (Path(__file__).resolve(),Path(training.__file__),WORKSPACE/'tools/verify_nearby_zero_study.py'):
            sources[str(path)]=ex.sha(path)
        ex.write(ROOT/'declaration.json',dict(**d,source_hashes=sources,reference_hashes=parent['reference_hashes'],created_at=datetime.now(timezone.utc).isoformat(),cv_seed=339,cv_updates=1200,arms=['nearby_zero'],
            parent_declaration=str(control.ROOT/'declaration.json'),parent_declaration_sha256=ex.sha(control.ROOT/'declaration.json'),
            intervention='For real-C legacy known-zero draws only, separate RNG seed339+32532 replaces index with probability0.5 by uniform existing trusted zero within43 frames of known positive. Use same replaced index for paired original. If pool empty retain old draw. Preserve legacy source/positive/feature/original-anchor draw schedule and all labels, weights, MSE response loss,51x8 Transformer,1200 updates,AdamW,seed339 and current renderer/guards. No unknown/ramp relabelling, new data, runtime label use or support alteration.',
            rationale='Descriptive audit shows near zeros2.22-2.82% of all draws; test spatial discrimination without claiming causality or guaranteed improvement. Radius43 prospectively fixed to existing minimum real-support scale.',
            control='Reuse four exactly reproduced V31 MSE controls authenticated against V12; no new control fits. Every legacy trace digest must match V12; separately save all actual query indices and independently reconstruct intervention.',
            selection='All-singer raw gate AND all50 originals identity AND strictly lower singer-macro delivered remaining error than reference20.3421417351c; count alone insufficient. No development or confirmation before qualified lock.',
            demo_activated=False,research_validation_passed=False))
        snapshot=ROOT/'executed_sources'; snapshot.mkdir(exist_ok=True)
        for path in sources:
            (snapshot/Path(path).name).write_bytes(Path(path).read_bytes())
        for name in ('train_pairs.json','anchor_sources.json'):
            (ROOT/name).write_bytes((control.ROOT/name).read_bytes())
    d=ex.read(ROOT/'declaration.json')
    for kind in ('source_hashes','input_hashes','reference_hashes'):
        for path,digest in d[kind].items():
            assert ex.sha(path)==digest,path
    assert ex.sha(d['parent_declaration'])==d['parent_declaration_sha256']
    for kind in ('baseline_cv','baseline_evidence'):
        assert ex.sha(d[kind])==d[kind+'_sha256']
    return d


def cross_validation(d):
    if (ROOT/'cross_validation.json').exists():
        raise ValueError('Retain completed nearby-zero study')
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks={pool.submit(training.cv_worker,(str(ROOT),f,'nearby_zero')):f for f in range(4)}
        for task in as_completed(tasks):
            result=task.result(); fold=tasks[task]+1; rows.extend(result['rows'])
            receipt=ex.read(ROOT/f'cross_validation/fold_{fold}/nearby_zero/fit_receipt.json')
            reference=ex.read(BASE/f'cross_validation/fold_{fold}/matched_baseline/fit_receipt.json')
            for key in ('sampling_sha256','training_sources','training_groups'):
                assert receipt[key]==reference[key],key
            assert receipt['response_metric']=='mse'
            print('NEARBY_ZERO_FOLD',fold,result['summary']['raw_mae_cents'],receipt['changed_query_rows'],flush=True)
    ex.write(ROOT/'cross_validation.json',dict(results={'nearby_zero':summarize(rows)},new_neural_fits=4,control_fits_reused=4))
    print('NEARBY_ZERO_CV_COMPLETE',flush=True)


def delivery(d):
    rows = ex.read(ROOT / 'train_pairs.json'); pairs = training.held.prepared(rows)
    models, cache = {}, {}
    for fold, groups in enumerate(d['folds'], 1):
        folder = ROOT / f'cross_validation/fold_{fold}/nearby_zero'
        model, payload = training.load(folder / 'checkpoint.pt')
        subset = [r for r in rows if ex.group(r) in groups]
        assert_independent(payload['training_groups'], payload['training_sources'], subset)
        assert payload['response_metric'] == 'mse'
        for group in groups:
            models[group] = model, payload
        with np.load(folder / 'validation/predictions.npz') as z:
            for i, row in enumerate(subset):
                if 'excluded' not in row:
                    for view in ('A', 'C'):
                        cache[row['path_' + view]] = z[f'{i}_{view}'].copy()
    folder = ROOT / 'arm'; folder.mkdir(exist_ok=True)
    if not (folder / 'declaration.json').exists():
        ex.write(folder / 'declaration.json', dict(parent=str(ROOT / 'declaration.json'), parent_sha256=ex.sha(ROOT / 'declaration.json')))
    oldroot = curve.ROOT; curve.ROOT = folder
    try:
        evidence = curve.run_condition(rows, pairs, cache, models, ex.read(ROOT / 'anchor_sources.json'), 'median11', 'training', ROOT / 'train_pairs.json')
    finally:
        curve.ROOT = oldroot
    raw = ex.read(ROOT / 'cross_validation.json')['results']['nearby_zero']; baseline = ex.read(d['baseline_evidence'])
    selected = bool(raw['prediction_gate_passed'] and evidence['gates']['original_controls_all_identity']
                    and evidence['summary']['delivered_after_mae_cents'] < baseline['summary']['delivered_after_mae_cents'])
    ex.write(ROOT / 'locked_selection.json', dict(arm='nearby_zero' if selected else 'matched_baseline',
             prediction_gate_passed=raw['prediction_gate_passed'], all_originals_identity=evidence['gates']['original_controls_all_identity'],
             default_seed=339, demo_activated=False, research_validation_passed=False))
    print('NEARBY_ZERO_SELECTION', selected, flush=True)


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['declare','cv','delivery'])
    args=parser.parse_args(); d=declare()
    if args.stage=='cv': cross_validation(d)
    elif args.stage=='delivery': delivery(d)
    declare()

if __name__=='__main__': main()
