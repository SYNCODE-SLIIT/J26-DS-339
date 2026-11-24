"""Prospectively declared dense residual adapter and actual-WAV comparison."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / 'source'))
from bpdr_input.experiments.learning import dense_residual_training as training
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.dense_residual_transformer import load, predict_raw
from bpdr_input.modeling.correction_accuracy import summarize, assert_independent
import evaluate_candidate_distribution_readout as previous
import run_transformer_curve_study as curve

ROOT = WORKSPACE / 'results/transformer_usability/2026-10-30_v26_dense_residual'
BASE = previous.previous.BASE


def declare():
    ROOT.mkdir(exist_ok=True)
    old = previous.declare()
    if not (ROOT / 'declaration.json').exists():
        inherited = ('folds', 'default_seed', 'policy', 'curve_condition', 'renderer', 'input_hashes',
                     'baseline_cv', 'baseline_cv_sha256', 'baseline_evidence', 'baseline_evidence_sha256')
        d = {k: old[k] for k in inherited}
        files = dict(old['source_hashes'])
        paths = [Path(__file__).resolve(), Path(training.__file__),
                 WORKSPACE / 'source/bpdr_input/modeling/dense_residual_transformer.py',
                 Path(training.held.__file__), Path(training.held.recovery.__file__),
                 WORKSPACE / 'source/bpdr_input/experiments/learning/joint_transformer_training.py',
                 WORKSPACE / 'source/bpdr_input/modeling/correction_accuracy.py']
        files.update({str(p): ex.sha(p) for p in paths})
        parents = {str(f): str(BASE / f'cross_validation/fold_{f}/matched_baseline/checkpoint.pt') for f in range(1, 5)}
        full = old['full_refits']
        d.update(created_at=datetime.now(timezone.utc).isoformat(), model_family='dense_residual_context',
                 architecture='Frozen JointTransformer with original center hidden state plus3 input-only center cues. Zero-initialized35-to1 amplitude-logit and activity-logit deltas72 parameters. Same bounded30*tanh amplitude and sigmoid activity. Dense per-frame inference; no pooled or constant-region override.',
                 control='Require bitwise zero-adapter/reference output equivalence on every unique training/feature-augmentation input before fitting, and every scored held-out input in independent verification. No new control backbone fit.',
                 training='Same base108 valid actual-audio pairs plus existing joint feature-only augmentations from reliable training originals. Identical source/singer partitions and old joint frame sampler/loss:12 triples/update;25% augmentation,70% plateau sampling;original weights4, outside2, positive1;anchor +0.25 response +0.5 BCE +0.25 amplitude. Frozen embeddings; only72 adapter parameters update.',
                 budget='AdamW lr0.001 weight_decay0.001, clip1,1200 head-only CV updates; conditional2400 full-head updates. Not matched-budget evidence of backbone/objective superiority.',
                 cv_seed=339, cv_updates=1200, final_seeds=[339, 340, 341], final_updates=2400,
                 cv_parents=parents, full_parents=full,
                 parent_hashes={p: ex.sha(p) for p in list(parents.values()) + list(full.values())},
                 source_hashes=files, previous_selection_sha256=ex.sha(previous.ROOT / 'locked_selection.json'),
                 selection='Raw all-singer prediction gate, all50 original identities, then lower singer-macro delivered error than reference. Default339 fixed. No threshold/support/renderer change. No known-development or confirmation access before qualification and independent verification.',
                 new_backbone_fits=0, demo_activated=False, research_validation_passed=False)
        ex.write(ROOT / 'declaration.json', d)
        snapshots = ROOT / 'executed_sources'; snapshots.mkdir(exist_ok=True)
        for path in files:
            (snapshots / Path(path).name).write_bytes(Path(path).read_bytes())
        for name in ('train_pairs.json', 'anchor_sources.json'):
            (ROOT / name).write_bytes((BASE / name).read_bytes())
    d = ex.read(ROOT / 'declaration.json')
    for label in ('source_hashes', 'input_hashes', 'parent_hashes'):
        for path, digest in d[label].items():
            assert ex.sha(path) == digest, path
    for label in ('baseline_cv', 'baseline_evidence'):
        assert ex.sha(d[label]) == d[label + '_sha256']
    assert ex.sha(previous.ROOT / 'locked_selection.json') == d['previous_selection_sha256']
    return d


def delivery(d):
    rows = ex.read(ROOT / 'train_pairs.json'); pairs = training.held.prepared(rows)
    cache, models = {}, {}
    for fold, groups in enumerate(d['folds'], 1):
        folder = ROOT / f'cross_validation/fold_{fold}/dense_residual'
        model, payload = load(folder / 'checkpoint.pt')
        subset = [r for r in rows if ex.group(r) in groups]
        assert_independent(payload['training_groups'], payload['training_sources'], subset)
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
    oldroot, oldpredict = curve.ROOT, curve.predict_raw
    curve.ROOT, curve.predict_raw = folder, predict_raw
    try:
        return curve.run_condition(rows, pairs, cache, models, ex.read(ROOT / 'anchor_sources.json'), 'median11', 'training', ROOT / 'train_pairs.json')
    finally:
        curve.ROOT, curve.predict_raw = oldroot, oldpredict


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['declare', 'cv', 'delivery', 'refit'])
    args = parser.parse_args(); d = declare()
    if args.stage == 'cv':
        rows = []
        with ProcessPoolExecutor(max_workers=2) as pool:
            tasks = {pool.submit(training.cv_worker, (str(ROOT), f)): f for f in range(4)}
            for task in as_completed(tasks):
                result = task.result(); rows.extend(result['rows'])
                print('DENSE_RESIDUAL_FOLD', tasks[task] + 1, result['summary']['raw_mae_cents'], flush=True)
        results = dict(matched_baseline=ex.read(d['baseline_cv'])['results']['matched_baseline'], dense_residual=summarize(rows))
        ex.write(ROOT / 'cross_validation.json', dict(results=results, new_head_fits=4, new_backbone_fits=0))
        print('DENSE_RESIDUAL_CV', results['dense_residual']['prediction_gate_passed'], results['dense_residual']['summary'], flush=True)
    elif args.stage == 'delivery':
        evidence = delivery(d); cv = ex.read(ROOT / 'cross_validation.json')['results']['dense_residual']
        baseline = ex.read(d['baseline_evidence'])
        selected = bool(cv['prediction_gate_passed'] and evidence['gates']['original_controls_all_identity']
                        and evidence['summary']['delivered_after_mae_cents'] < baseline['summary']['delivered_after_mae_cents'])
        ex.write(ROOT / 'locked_selection.json', dict(arm='dense_residual' if selected else 'matched_baseline',
                 prediction_gate_passed=cv['prediction_gate_passed'], all_originals_identity=evidence['gates']['original_controls_all_identity'],
                 default_seed=339, diagnostic_only=True, demo_activated=False, research_validation_passed=False))
        print('DENSE_RESIDUAL_SELECTION', selected, flush=True)
    elif args.stage == 'refit':
        choice = ex.read(ROOT / 'locked_selection.json')
        if choice['arm'] != 'dense_residual' or not choice['prediction_gate_passed'] or not choice['all_originals_identity']:
            raise ValueError('No qualified dense residual adapter selected; no refit')
        receipt = ex.read(ROOT / 'integrity_receipt.json')
        assert receipt['declaration_sha256'] == ex.sha(ROOT / 'declaration.json')
        for seed in d['final_seeds']:
            folder = ROOT / f'final/seed_{seed}'
            if (folder / 'fit_receipt.json').exists():
                continue
            training.fit(training.held.prepared(ex.read(ROOT / 'train_pairs.json')),
                         training.held.anchor_pool(ex.read(ROOT / 'anchor_sources.json')), seed, d['final_updates'],
                         folder, ex.sha(ROOT / 'declaration.json'), Path(d['full_parents'][str(seed)]))
            print('DENSE_RESIDUAL_REFIT', seed, flush=True)
    declare()


if __name__ == '__main__':
    main()
