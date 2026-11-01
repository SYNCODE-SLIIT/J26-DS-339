"""Replay independent model-family predictions and encoded outputs after delivery."""
import argparse
from pathlib import Path
import numpy as np
import torch
import compare_sequence_models as sequence
import compare_context_tree as classical
from bpdr_input.modeling.correction_accuracy import assert_independent
from verify_frozen_audio_studies import outputs


def verify(runner, families, loader, predict, extension):
    ex = runner.ex; d = runner.declare(); root = runner.ROOT
    rows = ex.read(root / 'train_pairs.json')
    pairs = sequence.training.held.prepared(rows)
    pair_map = {p['path_C']: p for p in pairs if 'excluded' not in p}
    global_indices = {r['path_C']: i for i, r in enumerate(rows) if 'excluded' not in r}
    sources = ex.read(root / 'anchor_sources.json'); reports = {}; hashes = {}
    for family in families:
        folder = root / 'arms' / family / 'training/median11' if runner is sequence else root / 'arm/training/median11'
        checked = 0
        with np.load(folder / 'predictions.npz') as delivered:
            for fold, groups in enumerate(d['folds'], 1):
                trained = root / f'cross_validation/fold_{fold}/{family}'
                checkpoint = trained / ('checkpoint.' + extension)
                model, payload = loader(checkpoint)
                receipt = ex.read(trained / 'fit_receipt.json')
                reference = ex.read(d['reference_receipts'][str(fold)])
                assert receipt['checkpoint_sha256'] == ex.sha(checkpoint) and receipt['reload_verified']
                assert payload['declaration_sha256'] == ex.sha(root / 'declaration.json')
                assert payload['sampling_sha256'] == receipt['sampling_sha256'] == reference['sampling_sha256']
                assert payload['training_groups'] == receipt['training_groups'] == reference['training_groups']
                assert payload['training_sources'] == receipt['training_sources'] == reference['training_sources']
                subset = [r for r in rows if ex.group(r) in groups]
                assert_independent(payload['training_groups'], payload['training_sources'], subset)
                anchors = sequence.training.held.anchor_pool(sources, groups)
                assert payload['original_anchor_recordings'] == len(anchors)
                assert payload['feature_only_training_examples'] == len(sequence.training.augmented_pairs(anchors, d['cv_seed']))
                if runner is sequence:
                    assert payload['family'] == family and payload['updates'] == 1200
                    assert receipt['parameters'] == sum(p.numel() for p in model.parameters())
                else:
                    assert payload['sampled_rows'] == 43200 and payload['sampler_updates'] == 1200
                    assert model.n_estimators == 256 and model.max_depth == 18 and model.min_samples_leaf == 8
                with np.load(trained / 'validation/predictions.npz') as cv:
                    for i, row in enumerate(subset):
                        if 'excluded' in row:
                            continue
                        pair = pair_map[row['path_C']]
                        for view in ('A', 'C'):
                            actual = predict(model, pair[view]['views'][0])[0]
                            # Tree reductions can differ at float64 summation roundoff;
                            # this replay tolerance is not an edit/audit threshold change.
                            np.testing.assert_allclose(actual, cv[f'{i}_{view}'][0], atol=1e-6, rtol=1e-7)
                            np.testing.assert_array_equal(cv[f'{i}_{view}'], delivered[f'{global_indices[row["path_C"]]}_{view}'])
                            np.testing.assert_array_equal(cv[f'{i}_{view}'][1:], np.stack([cv[f'{i}_{view}'][0]] * 2))
                            checked += 1
                hashes[str(checkpoint)] = ex.sha(checkpoint)
                print('MODEL_FAMILY_FOLD_VERIFIED', family, fold, flush=True)
        reports[family] = dict(**outputs(folder, root / 'train_pairs.json', sources), held_out_input_views_replayed=checked)
    protected = ex.read(root.parent / '2026-10-28_v7_guard/declaration.json')['protected_files']
    for path, digest in protected.items():
        assert ex.sha(path) == digest
    assert not (root.parent / 'active_model.json').exists()
    runner.declare()
    result = dict(results=reports, checkpoint_hashes=hashes, matched_training_sampler_verified=True,
                  protected_files_unchanged=len(protected), activation_absent=True,
                  declaration_sha256=ex.sha(root / 'declaration.json'),
                  selection_sha256=ex.sha(root / 'locked_selection.json'),
                  verifier_sha256=ex.sha(Path(__file__)), research_validation_passed=False)
    ex.write(root / 'integrity_receipt.json', result)
    print('MODEL_FAMILY_VERIFIED', result, flush=True)


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['sequence', 'tree'])
    args = parser.parse_args()
    if args.kind == 'sequence':
        verify(sequence, sequence.FAMILIES, sequence.alternative.load, sequence.alternative.predict_raw, 'pt')
    else:
        verify(classical, ('context_extra_trees',), classical.tree.load, classical.tree.predict_raw, 'joblib')


if __name__ == '__main__':
    main()
