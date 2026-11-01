"""Fresh ExtraTrees context baseline under matched source folds and sample trace."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import sklearn
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / 'source'))
from bpdr_input.experiments.learning import context_tree_training as training, context_improvement as ex
from bpdr_input.modeling import context_tree as tree
from bpdr_input.modeling.correction_accuracy import summarize, assert_independent
import compare_sequence_models as previous
import run_transformer_curve_study as curve

ROOT = WORKSPACE / 'results/transformer_usability/2026-10-30_v28_context_tree'


def declare():
    ROOT.mkdir(exist_ok=True)
    old = previous.declare()
    if not (ROOT / 'declaration.json').exists():
        keys = ('folds', 'default_seed', 'policy', 'curve_condition', 'renderer', 'input_hashes',
                'baseline_cv', 'baseline_cv_sha256', 'baseline_evidence', 'baseline_evidence_sha256',
                'reference_receipts', 'reference_receipt_hashes')
        d = {k: old[k] for k in keys}; files = dict(old['source_hashes'])
        for p in (Path(__file__).resolve(), Path(training.__file__), Path(tree.__file__)):
            files[str(p)] = ex.sha(p)
        d.update(created_at=datetime.now(timezone.utc).isoformat(), model_family='context_extra_trees',
                 cv_seed=339, sampler_updates=1200, sklearn_version=sklearn.__version__, source_hashes=files,
                 feature_contract='Flatten identical51x8 input-only joint context plus51 presence-padding indicators:459 features. No IDs, labels, musical note/key or provisional boundaries.',
                 training='Materialize exactly1200x12x3 sampled rows43200 with the existing neural sampler and original/positive weights4/2/1. Require every fold sampler/source/group digest to match frozen V12 receipts. ExtraTrees256,max_depth18,min_samples_leaf8,max_features1.0,random_state339,n_jobs2.',
                 limitation='Regression-only weighted squared error. No neural paired-response/activity/amplitude losses and no equal wall-clock or parameter budget claim. Targets bounded[-1,1] make each tree leaf and ensemble prediction bounded; no clipping. Activity return is None, not a confidence claim.',
                 selection='Same raw all-singer gate,50 original identities,then lower singer-macro delivered after error than frozen reference. All200 attempts/108 scoreable/92 exclusions retained. No numerical/renderer/support change or final-audio access.',
                 demo_activated=False, research_validation_passed=False)
        ex.write(ROOT / 'declaration.json', d)
        snapshot = ROOT / 'executed_sources'; snapshot.mkdir(exist_ok=True)
        for p in files:
            (snapshot / Path(p).name).write_bytes(Path(p).read_bytes())
        for name in ('train_pairs.json', 'anchor_sources.json'):
            (ROOT / name).write_bytes((previous.BASE / name).read_bytes())
    d = ex.read(ROOT / 'declaration.json')
    for label in ('source_hashes', 'input_hashes', 'reference_receipt_hashes'):
        for path, digest in d[label].items():
            assert ex.sha(path) == digest, path
    for label in ('baseline_cv', 'baseline_evidence'):
        assert ex.sha(d[label]) == d[label + '_sha256']
    assert sklearn.__version__ == d['sklearn_version']
    return d


def delivery(d):
    rows = ex.read(ROOT / 'train_pairs.json'); pairs = training.held.prepared(rows)
    models, cache = {}, {}
    for fold, groups in enumerate(d['folds'], 1):
        folder = ROOT / f'cross_validation/fold_{fold}/context_extra_trees'
        model, payload = tree.load(folder / 'checkpoint.joblib')
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
    curve.ROOT, curve.predict_raw = folder, tree.predict_raw
    try:
        return curve.run_condition(rows, pairs, cache, models, ex.read(ROOT / 'anchor_sources.json'), 'median11', 'training', ROOT / 'train_pairs.json')
    finally:
        curve.ROOT, curve.predict_raw = oldroot, oldpredict


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['declare', 'cv', 'delivery'])
    args = parser.parse_args(); d = declare()
    if args.stage == 'cv':
        rows = []
        with ProcessPoolExecutor(max_workers=2) as pool:
            tasks = {pool.submit(training.worker, (str(ROOT), fold)): fold for fold in range(4)}
            for task in as_completed(tasks):
                result = task.result(); rows.extend(result['rows'])
                print('CONTEXT_TREE_FOLD', tasks[task] + 1, result['summary']['raw_mae_cents'], flush=True)
        results = dict(context_extra_trees=summarize(rows), matched_baseline=ex.read(d['baseline_cv'])['results']['matched_baseline'])
        ex.write(ROOT / 'cross_validation.json', dict(results=results, new_tree_fits=4, matched_sampling_verified=True))
        print('CONTEXT_TREE_CV', results['context_extra_trees']['prediction_gate_passed'], results['context_extra_trees']['summary'], flush=True)
    elif args.stage == 'delivery':
        evidence = delivery(d); cv = ex.read(ROOT / 'cross_validation.json')['results']['context_extra_trees']
        baseline = ex.read(d['baseline_evidence'])
        selected = bool(cv['prediction_gate_passed'] and evidence['gates']['original_controls_all_identity']
                        and evidence['summary']['delivered_after_mae_cents'] < baseline['summary']['delivered_after_mae_cents'])
        ex.write(ROOT / 'locked_selection.json', dict(arm='context_extra_trees' if selected else 'matched_baseline',
                 prediction_gate_passed=cv['prediction_gate_passed'], all_originals_identity=evidence['gates']['original_controls_all_identity'],
                 default_seed=339, demo_activated=False, research_validation_passed=False))
        print('CONTEXT_TREE_SELECTION', selected, flush=True)
    declare()


if __name__ == '__main__':
    main()
