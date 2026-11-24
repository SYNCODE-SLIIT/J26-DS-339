"""Independent residual-parent, zero-control, prediction and saved-WAV checks."""
from pathlib import Path
import numpy as np
import torch
import train_dense_residual_transformer as study
from bpdr_input.modeling.dense_residual_transformer import DenseResidualTransformer, load, predict_raw
from bpdr_input.modeling.joint_transformer import load as parent_load
from bpdr_input.modeling.correction_accuracy import assert_independent
from verify_frozen_audio_studies import outputs


def main():
    torch.set_num_threads(2)
    d = study.declare(); ex = study.ex
    rows = ex.read(study.ROOT / 'train_pairs.json')
    pairs = study.training.held.prepared(rows)
    pair_map = {p['path_C']: p for p in pairs if 'excluded' not in p}
    sources = ex.read(study.ROOT / 'anchor_sources.json')
    checked = 0
    delivered_path = study.ROOT / 'arm/training/median11/predictions.npz'
    with np.load(delivered_path) as delivered:
        for fold, groups in enumerate(d['folds'], 1):
            folder = study.ROOT / f'cross_validation/fold_{fold}/dense_residual'
            model, payload = load(folder / 'checkpoint.pt')
            receipt = ex.read(folder / 'fit_receipt.json')
            assert receipt['checkpoint_sha256'] == ex.sha(folder / 'checkpoint.pt')
            assert receipt['reload_verified'] and receipt['initial_equivalence_exact']
            assert payload['declaration_sha256'] == ex.sha(study.ROOT / 'declaration.json')
            assert payload['parent_checkpoint_sha256'] == ex.sha(d['cv_parents'][str(fold)])
            assert payload['trainable_parameters'] == sum(p.numel() for p in model.parameters() if p.requires_grad) == 72
            parent, parent_payload = parent_load(d['cv_parents'][str(fold)])
            assert all(torch.equal(v, model.parent.state_dict()[k]) for k, v in parent.state_dict().items())
            assert payload['training_groups'] == parent_payload['training_groups']
            assert payload['training_sources'] == parent_payload['training_sources']
            anchors = study.training.held.anchor_pool(sources, groups)
            assert payload['original_anchor_recordings'] == len(anchors)
            assert payload['feature_only_training_examples'] == len(study.training.augmented_pairs(anchors, d['cv_seed']))
            initial = ex.read(folder / 'initial_equivalence.json')
            assert initial['exact'] and initial['maximum_error'] == 0.
            zero = DenseResidualTransformer(parent)
            subset = [r for r in rows if ex.group(r) in groups]
            assert_independent(payload['training_groups'], payload['training_sources'], subset)
            old_path = study.BASE / f'cross_validation/fold_{fold}/matched_baseline/validation/predictions.npz'
            with np.load(folder / 'validation/predictions.npz') as cv, np.load(old_path) as reference:
                for i, row in enumerate(subset):
                    if 'excluded' in row:
                        continue
                    pair = pair_map[row['path_C']]
                    global_index = next(j for j, r in enumerate(rows) if r.get('path_C') == row['path_C'])
                    for view in ('A', 'C'):
                        raw = pair[view]['views'][0]
                        np.testing.assert_array_equal(predict_raw(zero, raw)[0], reference[f'{i}_{view}'][0])
                        predicted = predict_raw(model, raw)[0]
                        np.testing.assert_array_equal(predicted, cv[f'{i}_{view}'][0])
                        np.testing.assert_array_equal(cv[f'{i}_{view}'], delivered[f'{global_index}_{view}'])
                        checked += 1
            print('DENSE_RESIDUAL_FOLD_VERIFIED', fold, flush=True)
    audio = outputs(study.ROOT / 'arm/training/median11', study.ROOT / 'train_pairs.json', sources)
    protected = ex.read(study.ROOT.parent / '2026-10-28_v7_guard/declaration.json')['protected_files']
    for path, digest in protected.items():
        assert ex.sha(path) == digest
    assert not (study.ROOT.parent / 'active_model.json').exists()
    study.declare()
    result = dict(**audio, frozen_parent_folds=4, initial_zero_control_exact=True,
                  held_out_input_views_replayed=checked, prediction_replay_exact=True,
                  protected_files_unchanged=len(protected), activation_absent=True,
                  declaration_sha256=ex.sha(study.ROOT / 'declaration.json'),
                  verifier_sha256=ex.sha(Path(__file__)), research_validation_passed=False)
    ex.write(study.ROOT / 'integrity_receipt.json', result)
    print('DENSE_RESIDUAL_VERIFIED', result, flush=True)


if __name__ == '__main__':
    main()
