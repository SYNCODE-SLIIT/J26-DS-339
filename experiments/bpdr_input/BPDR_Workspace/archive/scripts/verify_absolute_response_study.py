"""Verify matched control reproduction, objective lineage and delivered audio."""
from pathlib import Path
import numpy as np
import torch
import train_absolute_response_transformer as study
from verify_frozen_audio_studies import outputs


def main():
    torch.set_num_threads(2); d = study.declare(); ex = study.ex
    controls = [study.verify_control(f) for f in range(1, 5)]
    assert len(controls) == 4
    rows = ex.read(study.ROOT / 'train_pairs.json'); pairs = study.training.held.prepared(rows)
    pair_map = {p['path_C']: p for p in pairs if 'excluded' not in p}
    indices = {r['path_C']: i for i, r in enumerate(rows) if 'excluded' not in r}
    folder = study.ROOT / 'arm/training/median11'; replayed = 0; hashes = {}
    with np.load(folder / 'predictions.npz') as delivered:
        for fold, groups in enumerate(d['folds'], 1):
            trained = study.ROOT / f'cross_validation/fold_{fold}/absolute_response'
            model, payload = study.training.load(trained / 'checkpoint.pt')
            receipt = ex.read(trained / 'fit_receipt.json')
            reference = ex.read(study.BASE / f'cross_validation/fold_{fold}/matched_baseline/fit_receipt.json')
            assert payload['declaration_sha256'] == ex.sha(study.ROOT / 'declaration.json')
            assert receipt['checkpoint_sha256'] == ex.sha(trained / 'checkpoint.pt') and receipt['reload_verified']
            assert payload['response_metric'] == receipt['response_metric'] == 'mae'
            assert payload['positive_weight'] == 1. and payload['updates'] == 1200 and payload['seed'] == 339
            for key in ('sampling_sha256', 'training_groups', 'training_sources'):
                assert payload[key] == receipt[key] == reference[key]
            anchors = study.training.held.anchor_pool(ex.read(study.ROOT / 'anchor_sources.json'), groups)
            assert payload['original_anchor_recordings'] == len(anchors)
            assert payload['feature_only_training_examples'] == len(study.training.augmented_pairs(anchors, 339))
            subset = [r for r in rows if ex.group(r) in groups]
            study.assert_independent(payload['training_groups'], payload['training_sources'], subset)
            control = torch.load(study.ROOT / f'cross_validation/fold_{fold}/matched_baseline/checkpoint.pt', map_location='cpu', weights_only=True)
            assert any(not torch.equal(v, control['state'][k]) for k, v in model.state_dict().items())
            with np.load(trained / 'validation/predictions.npz') as z:
                for i, row in enumerate(subset):
                    if 'excluded' in row:
                        continue
                    pair = pair_map[row['path_C']]
                    for view in ('A', 'C'):
                        output = study.training.predict_raw(model, pair[view]['views'][0])[0]
                        np.testing.assert_array_equal(output, z[f'{i}_{view}'][0])
                        np.testing.assert_array_equal(z[f'{i}_{view}'], delivered[f'{indices[row["path_C"]]}_{view}'])
                        replayed += 1
            hashes[str(trained / 'checkpoint.pt')] = ex.sha(trained / 'checkpoint.pt')
            print('ABSOLUTE_RESPONSE_FOLD_VERIFIED', fold, flush=True)
    audio = outputs(folder, study.ROOT / 'train_pairs.json', ex.read(study.ROOT / 'anchor_sources.json'))
    evidence = ex.read(folder / 'evidence.json'); baseline = ex.read(d['baseline_evidence'])
    cv = ex.read(study.ROOT / 'cross_validation.json')['results']['absolute_response']
    choice = ex.read(study.ROOT / 'locked_selection.json')
    selected = bool(cv['prediction_gate_passed'] and evidence['gates']['original_controls_all_identity']
                    and evidence['summary']['delivered_after_mae_cents'] < baseline['summary']['delivered_after_mae_cents'])
    assert choice['arm'] == ('absolute_response' if selected else 'matched_baseline')
    protected = ex.read(study.ROOT.parent / '2026-10-28_v7_guard/declaration.json')['protected_files']
    for path, digest in protected.items():
        assert ex.sha(path) == digest
    assert not (study.ROOT.parent / 'active_model.json').exists()
    study.declare()
    receipt = dict(**audio, controls_reproduced_exact=4, objective_folds_verified=4,
                   prediction_views_replayed=replayed, checkpoint_hashes=hashes,
                   declaration_sha256=ex.sha(study.ROOT / 'declaration.json'),
                   selection_sha256=ex.sha(study.ROOT / 'locked_selection.json'),
                   evidence_sha256=ex.sha(folder / 'evidence.json'), verifier_sha256=ex.sha(Path(__file__)),
                   protected_files_unchanged=len(protected), selected=selected,
                   activation_absent=True, research_validation_passed=False)
    ex.write(study.ROOT / 'integrity_receipt.json', receipt)
    print('ABSOLUTE_RESPONSE_STUDY_VERIFIED', receipt, flush=True)


if __name__ == '__main__':
    main()
