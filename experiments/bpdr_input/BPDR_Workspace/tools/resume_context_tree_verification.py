"""Finish saved tree fits if a bitwise parallel-reduction reload check failed.

Run only after the original CV command is terminal. Retains frozen fit code and
checkpoints; records the post-fit verification amendment explicitly. No refit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import compare_context_tree as study
from bpdr_input.reporting.transformer_accuracy_report import complete_report


def recover_receipts(d):
    ex = study.ex; root = study.ROOT
    rows = ex.read(root / 'train_pairs.json'); sources = ex.read(root / 'anchor_sources.json')
    # Require every original fit to have reached checkpoint saving before replay.
    paths = [root / f'cross_validation/fold_{f}/context_extra_trees/checkpoint.joblib' for f in range(1, 5)]
    if not all(p.exists() for p in paths):
        raise ValueError('Original fitting is incomplete; retain and poll its existing process, do not refit')
    for fold, checkpoint in enumerate(paths, 1):
        folder = checkpoint.parent
        if (folder / 'fit_receipt.json').exists():
            continue
        model, payload = study.tree.load(checkpoint)
        assert payload['declaration_sha256'] == ex.sha(root / 'declaration.json')
        reference = ex.read(d['reference_receipts'][str(fold)])
        excluded = set(d['folds'][fold - 1])
        training = study.training.held.prepared([r for r in rows if ex.group(r) not in excluded])
        anchors = study.training.held.anchor_pool(sources, excluded)
        _, queries, targets, _, digest, aug = study.training.sampled_queries(training, anchors, d['cv_seed'], d['sampler_updates'])
        assert digest == payload['sampling_sha256'] == reference['sampling_sha256']
        assert len(queries) == payload['sampled_rows'] == 43200
        assert payload['training_groups'] == reference['training_groups']
        assert payload['training_sources'] == reference['training_sources']
        assert payload['feature_only_training_examples'] == aug
        assert payload['original_anchor_recordings'] == len(anchors)
        assert np.all(abs(targets) <= 1.) and len(model.estimators_) == 256
        raw = next(p for p in training if 'excluded' not in p)['C']['views'][0]
        features = study.tree.features(raw, np.arange(min(128, len(raw))))
        parallel = model.predict(features)
        original_jobs = model.n_jobs
        model.n_jobs = 1
        serial = model.predict(features)
        restored, _ = study.tree.load(checkpoint); restored.n_jobs = 1
        np.testing.assert_array_equal(serial, restored.predict(features))
        np.testing.assert_allclose(parallel, serial, atol=1e-14, rtol=1e-14)
        maximum = float(np.max(abs(parallel - serial)))
        note = dict(reason='Frozen runner used bitwise equality on threaded float64 tree summation. Serial checkpoint reload is bitwise exact; parallel summation agrees to normalized1e-14. No training, prediction rule, permission or encoded-audio gate changed.',
                    source_sha256=ex.sha(Path(__file__)), checkpoint_sha256=ex.sha(checkpoint),
                    normalized_parallel_serial_max_abs=maximum, serial_reload_exact=True,
                    parameter_n_jobs_preserved_in_checkpoint=original_jobs, original_fit_seconds_not_recorded=True)
        ex.write(folder / 'postfit_verification_amendment.json', note)
        metadata = {k: v for k, v in payload.items() if k not in ('model', 'version', 'feature_version')}
        ex.write(folder / 'fit_receipt.json', dict(**metadata, checkpoint_sha256=ex.sha(checkpoint),
                 reload_verified=True, recovered_after_postfit_check=True,
                 verification_amendment_sha256=ex.sha(folder / 'postfit_verification_amendment.json'),
                 parameters=model.set_params(n_jobs=original_jobs).get_params(),
                 realized_nodes=sum(t.tree_.node_count for t in model.estimators_)))
        print('TREE_POSTFIT_CHECK_RECOVERED', fold, maximum, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-cv-terminal', action='store_true', required=True)
    parser.parse_args()
    d = study.declare(); recover_receipts(d); rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks = {pool.submit(study.training.worker, (str(study.ROOT), f)): f for f in range(4)}
        for task in as_completed(tasks):
            result = task.result(); rows.extend(result['rows'])
            print('TREE_EVALUATION_COMPLETE', tasks[task] + 1, flush=True)
    results = dict(context_extra_trees=study.summarize(rows), matched_baseline=study.ex.read(d['baseline_cv'])['results']['matched_baseline'])
    study.ex.write(study.ROOT / 'cross_validation.json', dict(results=results, new_tree_fits=4,
                   matched_sampling_verified=True, postfit_verification_amendment=True))
    for fold in range(1, 5):
        folder = study.ROOT / f'cross_validation/fold_{fold}/context_extra_trees/validation'
        result = study.ex.read(folder / 'accuracy.json')
        study.ex.write_report(folder, result, 'Context ExtraTrees acoustic correction benchmark')
        complete_report(folder)
    study.declare()
    print('CONTEXT_TREE_CV_RECOVERED', results['context_extra_trees']['prediction_gate_passed'], results['context_extra_trees']['summary'], flush=True)


if __name__ == '__main__':
    main()
