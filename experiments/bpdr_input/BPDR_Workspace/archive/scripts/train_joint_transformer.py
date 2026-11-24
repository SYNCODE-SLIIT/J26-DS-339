"""Run the declared joint-context comparison without touching demo selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import joint_transformer_training as training
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.correction_accuracy import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=WORKSPACE/'results/transformer_usability/2026-10-28_v8_joint')
    parser.add_argument('--phase', choices=['cv', 'refit'], default='cv')
    args = parser.parse_args()
    root = args.run_root.resolve()
    d = ex.read(root/'declaration.json')
    files = [Path(__file__), WORKSPACE/'source/bpdr_input/modeling/joint_transformer.py',
             WORKSPACE/'source/bpdr_input/experiments/learning/joint_transformer_training.py',
             WORKSPACE/'source/bpdr_input/modeling/held_voice_transformer.py',
             WORKSPACE/'source/bpdr_input/modeling/step_context.py',
             WORKSPACE/'source/bpdr_input/repair/held_voice_policy.py']
    if not (root/'training_source_hashes.json').exists():
        snapshot = root/'executed_training_sources'
        snapshot.mkdir(exist_ok=True)
        for p in files:
            (snapshot/p.name).write_bytes(p.read_bytes())
        ex.write(root/'training_source_hashes.json', {str(p): ex.sha(p) for p in files})
        hashes = ex.input_hashes(ex.read(root/'train_pairs.json'))
        for row in ex.read(root/'anchor_sources.json'):
            if 'working_audio' in row:
                path = Path(row['working_audio'])
                for p in (path, path.parent/'original_features.npz'):
                    hashes[str(p)] = ex.sha(p)
        ex.write(root/'training_input_hashes.json', hashes)
    for name in ('training_source_hashes', 'training_input_hashes'):
        for p, h in ex.read(root/f'{name}.json').items():
            if ex.sha(p) != h:
                raise ValueError(f'Frozen training input/source changed: {p}')
    if args.phase == 'cv':
        rows = {arm: [] for arm in d['arms']}
        with ProcessPoolExecutor(max_workers=2) as pool:
            tasks = {pool.submit(training.cv_worker, (str(root), fold, arm)): (fold, arm)
                     for arm in d['arms'] for fold in range(len(d['folds']))}
            for future in as_completed(tasks):
                fold, arm = tasks[future]
                result = future.result()
                rows[arm].extend(result['rows'])
                print('JOINT_CV_COMPLETE', arm, fold+1, result['summary']['raw_mae_cents'], flush=True)
        results = {arm: summarize(rr) for arm, rr in rows.items()}
        selected = min(d['arms'], key=lambda arm: (not results[arm]['prediction_gate_passed'],
                       results[arm]['summary']['balanced_error_cents'], d['arms'].index(arm)))
        ex.write(root/'locked_selection.json', dict(arm=selected, results=results,
                 default_seed=d['default_seed'], prediction_gate_passed=results[selected]['prediction_gate_passed'],
                 diagnostic_only=True, selection_data='training-side singer held-out CV only',
                 research_validation_passed=False, demo_activated=False))
        print('JOINT_CV_LOCKED', selected, {a: v['summary'] for a, v in results.items()}, flush=True)
    else:
        selection = ex.read(root/'locked_selection.json')
        if not selection['prediction_gate_passed']:
            raise ValueError('No all-singer prediction pass; diagnose before final refits')
        pairs = training.held.prepared(ex.read(root/'train_pairs.json'))
        anchors = training.held.anchor_pool(ex.read(root/'anchor_sources.json'))
        for seed in d['final_seeds']:
            folder = root/f'final/seed_{seed}'
            if (folder/'fit_receipt.json').exists():
                continue
            training.fit(pairs, anchors, selection['arm'], seed, d['final_updates'], folder, ex.sha(root/'declaration.json'))
            print('JOINT_REFIT_COMPLETE', seed, flush=True)


if __name__ == '__main__':
    main()
