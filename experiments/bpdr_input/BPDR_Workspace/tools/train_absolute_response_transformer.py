"""Matched MSE-versus-absolute counter-detuning response; fixed output checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / 'source'))
from bpdr_input.experiments.learning import absolute_response_training as training
from bpdr_input.modeling.correction_accuracy import summarize, assert_independent
import train_dense_residual_transformer as previous
import run_transformer_curve_study as curve

ex = previous.ex
ROOT = previous.ROOT.parent / '2026-11-01_v31_absolute_response'
BASE = previous.BASE
ARMS = ('matched_baseline', 'absolute_response')


def declare():
    ROOT.mkdir(exist_ok=True); old = previous.declare()
    if not (ROOT / 'declaration.json').exists():
        retained = ('folds', 'default_seed', 'policy', 'curve_condition', 'renderer', 'input_hashes',
                    'baseline_cv', 'baseline_cv_sha256', 'baseline_evidence', 'baseline_evidence_sha256')
        d = {k: old[k] for k in retained}; sources = dict(old['source_hashes'])
        for path in (Path(__file__).resolve(), Path(training.__file__),
                     WORKSPACE / 'source/bpdr_input/experiments/learning/response_transformer_training.py'):
            sources[str(path)] = ex.sha(path)
        controls = [BASE / f'cross_validation/fold_{f}/matched_baseline' for f in range(1, 5)]
        control_hashes = {str(p / name): ex.sha(p / name) for p in controls
                          for name in ('checkpoint.pt', 'fit_receipt.json', 'validation/predictions.npz')}
        ex.write(ROOT / 'declaration.json', dict(**d, created_at=datetime.now(timezone.utc).isoformat(),
                 source_hashes=sources, reference_hashes=control_hashes, arms=list(ARMS),
                 cv_seed=339, cv_updates=1200, final_seeds=[339, 340, 341], final_updates=2400,
                 architecture='Unchanged51x8 joint-context Transformer,32 hidden,2 layers,4 heads,96 FF, existing amplitude*sigmoid(activity) output. Train from scratch for both arms.',
                 intervention='Change ONLY normalized paired response residual from mean square to mean absolute; coefficient0.25 unchanged. Keep weighted squared anchors, activity BCE0.5,positive-only amplitude MSE0.25,original/outside weights4/2 and positive weight1. Same data, augmentations,sampler,AdamW0.001,weight_decay0.001,gradient_clip1,1200 updates and seed.',
                 control='First reproduce all4 MSE control fits tensor-for-tensor, all validation prediction arrays and actual sampling digests against authenticated V12. Do not start MAE fits unless all controls match. Then require matching sampler/source/group receipts for every MAE fold. Reuse authenticated control WAV evidence only after exact equivalence.',
                 research_scope='Tests proposal-style mean-absolute counter-detuning response only. Not the full BPDR/augmentation-only/frame-only study, learned boundary-invariance claim or completion of RES-01; invariance remains input construction.',
                 selection='All-singer raw gate,all50 original WAVs identity,then strictly lower singer-macro delivered after error than matched reference. No repaired-count-only selection or threshold/support/renderer change. Default339 fixed; known development/confirmation only after qualified lock and independent verification.',
                 demo_activated=False, research_validation_passed=False))
        snapshot = ROOT / 'executed_sources'; snapshot.mkdir(exist_ok=True)
        for path in sources:
            (snapshot / Path(path).name).write_bytes(Path(path).read_bytes())
        for name in ('train_pairs.json', 'anchor_sources.json'):
            (ROOT / name).write_bytes((previous.ROOT / name).read_bytes())
    d = ex.read(ROOT / 'declaration.json')
    for kind in ('source_hashes', 'input_hashes', 'reference_hashes'):
        for path, digest in d[kind].items():
            assert ex.sha(path) == digest, path
    for kind in ('baseline_cv', 'baseline_evidence'):
        assert ex.sha(d[kind]) == d[kind + '_sha256']
    return d


def verify_control(fold):
    folder = ROOT / f'cross_validation/fold_{fold}/matched_baseline'
    old = BASE / f'cross_validation/fold_{fold}/matched_baseline'
    a = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
    b = torch.load(old / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(v, b['state'][k]) for k, v in a['state'].items()), 'MSE control weights differ'
    for key in ('training_groups', 'training_sources', 'sampling_sha256', 'seed', 'updates'):
        assert a[key] == b[key], key
    with np.load(folder / 'validation/predictions.npz') as x, np.load(old / 'validation/predictions.npz') as y:
        assert set(x.files) == set(y.files)
        for key in x.files:
            np.testing.assert_array_equal(x[key], y[key])
    return dict(fold=fold, parameters_exact=True, all_prediction_arrays_exact=True,
                sampling_sha256=a['sampling_sha256'], checkpoint_sha256=ex.sha(folder / 'checkpoint.pt'))


def cross_validation(d):
    if (ROOT / 'cross_validation.json').exists():
        raise ValueError('Retain completed absolute-response CV')
    collected = {}; controls = []
    for arm in ARMS:
        rows = []
        with ProcessPoolExecutor(max_workers=2) as pool:
            tasks = {pool.submit(training.cv_worker, (str(ROOT), f, arm)): f for f in range(4)}
            for task in as_completed(tasks):
                result = task.result(); fold = tasks[task] + 1; rows.extend(result['rows'])
                if arm == 'matched_baseline':
                    controls.append(verify_control(fold))
                else:
                    actual = ex.read(ROOT / f'cross_validation/fold_{fold}/{arm}/fit_receipt.json')
                    reference = ex.read(BASE / f'cross_validation/fold_{fold}/matched_baseline/fit_receipt.json')
                    for key in ('sampling_sha256', 'training_groups', 'training_sources'):
                        assert actual[key] == reference[key], key
                    assert actual['response_metric'] == 'mae'
                print('ABSOLUTE_RESPONSE_FOLD', arm, fold, result['summary']['raw_mae_cents'], flush=True)
        collected[arm] = summarize(rows)
        if arm == 'matched_baseline':
            assert len(controls) == 4
            ex.write(ROOT / 'control_equivalence.json', dict(folds=controls, all_exact=True))
    ex.write(ROOT / 'cross_validation.json', dict(results=collected, new_neural_fits=8,
             four_controls_exact=True, paired_response_only_intervention=True))
    print('ABSOLUTE_RESPONSE_CV', [(k, v['prediction_gate_passed'], v['summary']['raw_mae_cents']) for k, v in collected.items()], flush=True)


def delivery(d):
    assert ex.read(ROOT / 'control_equivalence.json')['all_exact']
    rows = ex.read(ROOT / 'train_pairs.json'); pairs = training.held.prepared(rows)
    models, cache = {}, {}
    for fold, groups in enumerate(d['folds'], 1):
        folder = ROOT / f'cross_validation/fold_{fold}/absolute_response'
        model, payload = training.load(folder / 'checkpoint.pt')
        subset = [r for r in rows if ex.group(r) in groups]
        assert_independent(payload['training_groups'], payload['training_sources'], subset)
        assert payload['response_metric'] == 'mae'
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
    raw = ex.read(ROOT / 'cross_validation.json')['results']['absolute_response']; baseline = ex.read(d['baseline_evidence'])
    selected = bool(raw['prediction_gate_passed'] and evidence['gates']['original_controls_all_identity']
                    and evidence['summary']['delivered_after_mae_cents'] < baseline['summary']['delivered_after_mae_cents'])
    ex.write(ROOT / 'locked_selection.json', dict(arm='absolute_response' if selected else 'matched_baseline',
             prediction_gate_passed=raw['prediction_gate_passed'], all_originals_identity=evidence['gates']['original_controls_all_identity'],
             default_seed=339, demo_activated=False, research_validation_passed=False))
    print('ABSOLUTE_RESPONSE_SELECTION', selected, flush=True)


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['declare', 'cv', 'delivery'])
    args = parser.parse_args(); d = declare()
    if args.stage == 'cv':
        cross_validation(d)
    elif args.stage == 'delivery':
        delivery(d)
    declare()


if __name__ == '__main__':
    main()
