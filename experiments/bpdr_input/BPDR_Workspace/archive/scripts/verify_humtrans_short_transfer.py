"""Reload all eight checkpoints; replay holdouts, sampler traces and predictions."""
import numpy as np
import torch

import compare_humtrans_short_transfer as run


def main():
    torch.set_num_threads(2)
    d, rows, originals = run.declare()
    result = run.read(run.ROOT / 'results.json')
    assert result['declaration_sha256'] == run.sha(run.ROOT / 'declaration.json')
    for path, expected in d['source_hashes'].items():
        assert run.sha(path) == expected
    data = run.paired.records(rows, originals)
    predictions = {}
    for arm in d['arms']:
        with np.load(run.ROOT / (arm + '_all_predictions.npz')) as saved:
            predictions[arm] = {key: saved[key].copy() for key in saved.files}
        assert set(predictions[arm]) == {r['id'] for r in data}
    models = 0; arrays = 0; traces = 0
    for fold, held in enumerate(d['folds'], 1):
        train = [r for r in data if r['group'] not in held]
        test = [r for r in data if r['group'] in held]
        assert not {r['source_id'] for r in train} & {r['source_id'] for r in test}
        trace, digest = run.paired.previous.sample_trace(train, d)
        np.testing.assert_array_equal(trace, np.load(run.ROOT / f'fold_{fold}/physical_trace.npy'))
        fit_receipt = run.read(run.ROOT / f'fold_{fold}/fit_receipt.json')
        assert fit_receipt['physical_trace_sha256'] == digest
        assert fit_receipt['heldout_groups'] == held
        traces += 1
        for arm in d['arms']:
            path = (run.PARENT / f'fold_{fold}/paired_physical/checkpoint.pt' if arm == 'frozen_vocadito_physical'
                    else run.ROOT / f'fold_{fold}/checkpoint.pt')
            model, meta = run.paired.model_module.load(path)
            expected_train = [r for r in train if not r['group'].startswith('HumTrans:')] if arm == 'frozen_vocadito_physical' else train
            assert meta['training_groups'] == sorted({r['group'] for r in expected_train})
            assert meta['training_sources'] == sorted({r['source_id'] for r in expected_train})
            assert not set(meta['training_groups']) & set(held)
            assert not set(meta['training_sources']) & {r['source_id'] for r in test}
            if arm == 'added_real_hums':
                assert fit_receipt['checkpoint_sha256'] == run.sha(path)
                assert meta['declaration_sha256'] == run.sha(run.ROOT / 'declaration.json')
            else:
                assert d['parent_checkpoints'][str(path)] == run.sha(path)
            for record in test:
                p = run.paired.previous.predict(model, record, 'short_transformer')
                np.testing.assert_array_equal(p, predictions[arm][record['id']])
                arrays += 1
            models += 1
        print('HUM_TRANSFER_REPLAY_FOLD', fold, flush=True)
    for arm, pred in predictions.items():
        for name, subset in [('all', data), ('humming', [r for r in data if r['group'].startswith('HumTrans:')]),
                             ('vocadito', [r for r in data if not r['group'].startswith('HumTrans:')])]:
            summary = run.paired.previous.summarize(subset, pred)
            assert summary == result['arms'][arm][name]
            # Check reported per-group errors directly against held-out frame arrays.
            for group in summary['groups']:
                errors = []
                for r in subset:
                    if r['group'] != group['group'] or r['kind'] != 'corrupted':
                        continue
                    positive = r['mask'] & (r['target'] != 0)
                    if positive.any():
                        errors.append(float(np.mean(abs(pred[r['id']][positive] - r['target'][positive]))))
                assert len(errors) == group['scoreable_pairs']
                if errors:
                    assert float(np.mean(errors)) == group['positive_mae_cents']
    assert result['physical_attempts'] == 528 and result['scoreable_pairs'] == 176
    assert result['original_attempts'] == 33 and result['original_controls'] == 28
    expected_missing = sorted({r['group'] for r in originals} - {r['group'] for r in rows if 'excluded' not in r})
    assert result['groups_without_positive_pairs'] == expected_missing
    run.write(run.ROOT / 'integrity_receipt.json', dict(checkpoints_reloaded=models, prediction_arrays_reinferred=arrays,
        physical_sampler_traces_recreated=traces, source_singer_overlap_found=False,
        all_summaries_reproduced=True, groups_without_positive_pairs=expected_missing,
        results_sha256=run.sha(run.ROOT / 'results.json'), verifier_source_sha256=run.sha(__file__),
        raw_predictions_only=True, delivered_audio_qualification=False, production_eligible=False,
        research_validation_passed=False))
    print('HUM_TRANSFER_INDEPENDENT_REPLAY_PASS', models, arrays, traces, flush=True)


if __name__ == '__main__':
    main()
