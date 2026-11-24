"""Replay fixed activity decisions, reference equivalence and encoded WAVs."""
from pathlib import Path
import numpy as np
import torch
import evaluate_activity_decoding as study
from bpdr_input.modeling.correction_accuracy import summarize
from verify_frozen_audio_studies import outputs


def main():
    torch.set_num_threads(2)
    d = study.declare(); ex = study.ex
    rows = ex.read(study.ROOT / 'train_pairs.json'); pairs = study.previous.training.held.prepared(rows)
    model_map = study.models_for(d, rows)
    pair_map = {p['path_C']: p for p in pairs if 'excluded' not in p}
    indices = {r['path_C']: i for i, r in enumerate(rows) if 'excluded' not in r}
    cv = ex.read(study.ROOT / 'cross_validation.json')['results']['binary']
    assert summarize(cv['rows'])['prediction_gate_passed'] == cv['prediction_gate_passed']
    seen = set(); views = 0
    folder = study.ROOT / 'arm/training/median11'
    with np.load(study.ROOT / 'validation/predictions.npz') as predicted, np.load(folder / 'predictions.npz') as delivered:
        for fold, groups in enumerate(d['folds'], 1):
            subset = [r for r in rows if ex.group(r) in groups]
            with np.load(study.BASE / f'cross_validation/fold_{fold}/matched_baseline/validation/predictions.npz') as baseline:
                for i, row in enumerate(subset):
                    if 'excluded' in row:
                        continue
                    pair = pair_map[row['path_C']]; model = model_map[ex.group(row)][0]
                    for view in ('A', 'C'):
                        raw = pair[view]['views'][0]
                        original = study.decoding.predict_raw(model, raw, mode='expected')[0]
                        binary = study.decoding.predict_raw(model, raw)[0]
                        np.testing.assert_array_equal(original, baseline[f'{i}_{view}'][0])
                        key = f'{indices[row["path_C"]]}_{view}'
                        np.testing.assert_array_equal(binary, predicted[key][0])
                        np.testing.assert_array_equal(predicted[key], delivered[key])
                        np.testing.assert_array_equal(predicted[key][1:], np.stack([binary] * 2))
                        seen.add(pair['path_' + view]); views += 1
            print('ACTIVITY_DECODER_FOLD_VERIFIED', fold, flush=True)
    assert len(seen) == ex.read(study.ROOT / 'cross_validation.json')['expected_reference_unique_tracks_verified'] == 135
    audio = outputs(folder, study.ROOT / 'train_pairs.json', ex.read(study.ROOT / 'anchor_sources.json'))
    evidence = ex.read(folder / 'evidence.json'); baseline = ex.read(d['baseline_evidence'])
    choice = ex.read(study.ROOT / 'locked_selection.json')
    selected = bool(cv['prediction_gate_passed'] and evidence['gates']['original_controls_all_identity']
                    and evidence['summary']['delivered_after_mae_cents'] < baseline['summary']['delivered_after_mae_cents'])
    assert choice['arm'] == ('binary' if selected else 'matched_baseline')
    protected = ex.read(study.ROOT.parent / '2026-10-28_v7_guard/declaration.json')['protected_files']
    for path, digest in protected.items():
        assert ex.sha(path) == digest
    assert not (study.ROOT.parent / 'active_model.json').exists()
    study.declare()
    receipt = dict(**audio, prediction_views_replayed=views, unique_reference_tracks_exact=135,
                   frozen_cv_checkpoints=4, binary_selected=selected, protected_files_unchanged=len(protected),
                   declaration_sha256=ex.sha(study.ROOT / 'declaration.json'),
                   selection_sha256=ex.sha(study.ROOT / 'locked_selection.json'),
                   evidence_sha256=ex.sha(folder / 'evidence.json'), verifier_sha256=ex.sha(Path(__file__)),
                   activation_absent=True, research_validation_passed=False)
    ex.write(study.ROOT / 'integrity_receipt.json', receipt)
    print('ACTIVITY_DECODER_VERIFIED', receipt, flush=True)


if __name__ == '__main__':
    main()
