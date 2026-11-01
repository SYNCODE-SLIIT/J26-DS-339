"""Fixed expected-versus-binary activity decision; frozen Transformer and WAV gates."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / 'source'))
from bpdr_input.modeling import activity_decoding as decoding
from bpdr_input.modeling.joint_transformer import load
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.reporting.transformer_accuracy_report import complete_report
import train_dense_residual_transformer as previous
import run_transformer_curve_study as curve

ex = previous.ex
ROOT = previous.ROOT.parent / '2026-11-01_v30_activity_decoding'
BASE = previous.BASE


def declare():
    ROOT.mkdir(exist_ok=True); old = previous.declare()
    if not (ROOT / 'declaration.json').exists():
        keys = ('folds', 'default_seed', 'policy', 'curve_condition', 'renderer', 'input_hashes',
                'baseline_cv', 'baseline_cv_sha256', 'baseline_evidence', 'baseline_evidence_sha256',
                'cv_parents', 'full_parents', 'parent_hashes')
        d = {k: old[k] for k in keys}; sources = dict(old['source_hashes'])
        for path in (Path(__file__).resolve(), Path(decoding.__file__)):
            sources[str(path)] = ex.sha(path)
        predictions = {str(BASE / f'cross_validation/fold_{fold}/matched_baseline/validation/predictions.npz'):
                       ex.sha(BASE / f'cross_validation/fold_{fold}/matched_baseline/validation/predictions.npz') for fold in range(1, 5)}
        ex.write(ROOT / 'declaration.json', dict(**d, created_at=datetime.now(timezone.utc).isoformat(),
                 source_hashes=sources, prediction_hashes=predictions, decoder_version=decoding.VERSION,
                 rule='Expected reference outputs amplitude*sigmoid(activity_logit). Binary ablation outputs the SAME bounded conditional amplitude only where activity_logit>=0, otherwise zero; fixed MAP threshold0 (sigmoid0.5), repair wins tie. No fitting, amplitude clipping, per-case routing or threshold search. Unvoiced output zero. Activity score is NOT calibrated confidence.',
                 mechanism='Amplitude already has a separate positive-only supervised amplitude loss; activity has binary BCE. Test whether multiplication shrinks a correct conditional amount. This is an engineering inference ablation, not proof of objective superiority or trustworthy confidence.',
                 selection='All-singer raw gate,all50 original identities,then strictly lower singer-macro delivered after MAE than reference. Existing activation/release, real support, gain, protection, uncertainty and encoded-WAV checks unchanged. No development or confirmation access before locked qualification.',
                 new_neural_fits=0, demo_activated=False, research_validation_passed=False))
        snapshot = ROOT / 'executed_sources'; snapshot.mkdir(exist_ok=True)
        for path in sources:
            (snapshot / Path(path).name).write_bytes(Path(path).read_bytes())
        for name in ('train_pairs.json', 'anchor_sources.json'):
            (ROOT / name).write_bytes((previous.ROOT / name).read_bytes())
    d = ex.read(ROOT / 'declaration.json')
    for kind in ('source_hashes', 'input_hashes', 'parent_hashes', 'prediction_hashes'):
        for path, digest in d[kind].items():
            assert ex.sha(path) == digest, path
    for kind in ('baseline_cv', 'baseline_evidence'):
        assert ex.sha(d[kind]) == d[kind + '_sha256']
    return d


def models_for(d, rows):
    models = {}
    for fold, groups in enumerate(d['folds'], 1):
        model, payload = load(d['cv_parents'][str(fold)])
        assert_independent(payload['training_groups'], payload['training_sources'], [r for r in rows if ex.group(r) in groups])
        for group in groups:
            models[group] = model, payload
    return models


def cv(d):
    folder = ROOT / 'validation'; folder.mkdir(exist_ok=True)
    if (folder / 'accuracy.json').exists():
        raise ValueError('Retain completed activity comparison')
    rows = ex.read(ROOT / 'train_pairs.json'); pairs = previous.training.held.prepared(rows)
    pair_map = {p['path_C']: p for p in pairs if 'excluded' not in p}
    models = models_for(d, rows); cache = {}; verified = set()
    for fold, groups in enumerate(d['folds'], 1):
        subset = [r for r in rows if ex.group(r) in groups]
        path = BASE / f'cross_validation/fold_{fold}/matched_baseline/validation/predictions.npz'
        with np.load(path) as z:
            for i, row in enumerate(subset):
                if 'excluded' in row:
                    continue
                pair = pair_map[row['path_C']]; model = models[ex.group(row)][0]
                for view in ('A', 'C'):
                    key = pair['path_' + view]
                    if key not in verified:
                        original = decoding.predict_raw(model, pair[view]['views'][0], mode='expected')[0]
                        np.testing.assert_array_equal(original, z[f'{i}_{view}'][0])
                        cache[key] = np.stack([decoding.predict_raw(model, pair[view]['views'][0])[0]] * 3)
                        verified.add(key)
                    else:
                        # Repeated originals are not independent extra recordings.
                        np.testing.assert_array_equal(z[f'{i}_{view}'][1:], np.stack([z[f'{i}_{view}'][0]] * 2))
    ex.write(folder / 'evaluation_declaration.json', dict(pairs_manifest=str(ROOT / 'train_pairs.json'),
             role='Training-side held-singer/source fixed binary activity diagnostic, not fresh', decoder=decoding.VERSION))
    old = ex.predictions, ex.legacy_proposal
    ex.predictions = lambda model, pair, view: cache[pair['path_' + view]]
    ex.legacy_proposal = lambda pair, view, values: curve.conditioned_proposal(pair, view, values, curve.POLICY, 'median11')
    try:
        result = ex.evaluate(None, dict(seed=339), pairs, folder, render=False, seed=339)
    finally:
        ex.predictions, ex.legacy_proposal = old
    result.update(model_family='joint_binary_activity', decoder_version=decoding.VERSION)
    ex.write(folder / 'accuracy.json', result)
    ex.write_report(folder, result, 'Frozen Transformer binary activity ablation'); complete_report(folder)
    ex.write(ROOT / 'cross_validation.json', dict(results=dict(binary=result,
             matched_baseline=ex.read(d['baseline_cv'])['results']['matched_baseline']),
             expected_reference_unique_tracks_verified=len(verified), new_neural_fits=0))
    print('BINARY_ACTIVITY_CV', result['prediction_gate_passed'], result['summary'], flush=True)


def delivery(d):
    rows = ex.read(ROOT / 'train_pairs.json'); pairs = previous.training.held.prepared(rows)
    models = models_for(d, rows); cache = {}
    with np.load(ROOT / 'validation/predictions.npz') as z:
        for i, row in enumerate(rows):
            if 'excluded' not in row:
                for view in ('A', 'C'):
                    cache[row['path_' + view]] = z[f'{i}_{view}'].copy()
    folder = ROOT / 'arm'; folder.mkdir(exist_ok=True)
    if not (folder / 'declaration.json').exists():
        ex.write(folder / 'declaration.json', dict(parent=str(ROOT / 'declaration.json'), parent_sha256=ex.sha(ROOT / 'declaration.json')))
    oldroot, oldpredict = curve.ROOT, curve.predict_raw
    curve.ROOT, curve.predict_raw = folder, decoding.predict_raw
    try:
        evidence = curve.run_condition(rows, pairs, cache, models, ex.read(ROOT / 'anchor_sources.json'), 'median11', 'training', ROOT / 'train_pairs.json')
    finally:
        curve.ROOT, curve.predict_raw = oldroot, oldpredict
    raw = ex.read(ROOT / 'cross_validation.json')['results']['binary']; baseline = ex.read(d['baseline_evidence'])
    selected = bool(raw['prediction_gate_passed'] and evidence['gates']['original_controls_all_identity']
                    and evidence['summary']['delivered_after_mae_cents'] < baseline['summary']['delivered_after_mae_cents'])
    ex.write(ROOT / 'locked_selection.json', dict(arm='binary' if selected else 'matched_baseline',
             prediction_gate_passed=raw['prediction_gate_passed'], all_originals_identity=evidence['gates']['original_controls_all_identity'],
             default_seed=339, diagnostic_only=True, demo_activated=False, research_validation_passed=False))
    print('BINARY_ACTIVITY_SELECTION', selected, flush=True)


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['declare', 'cv', 'delivery'])
    args = parser.parse_args(); d = declare()
    if args.stage == 'cv':
        cv(d)
    elif args.stage == 'delivery':
        delivery(d)
    declare()


if __name__ == '__main__':
    main()
