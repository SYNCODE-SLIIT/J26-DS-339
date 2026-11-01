"""Replay fixed rank readouts and independently check their delivered WAVs."""
from pathlib import Path
import numpy as np
import torch

import evaluate_candidate_distribution_readout as study
from bpdr_input.experiments.learning import context_improvement as ex, held_voice_training as held
from bpdr_input.modeling import candidate_distribution_projection as readout
from bpdr_input.modeling.step_candidates import measured_candidates
from verify_frozen_audio_studies import outputs


def main():
    torch.set_num_threads(2)
    d = study.declare()
    rows = ex.read(study.ROOT / 'train_pairs.json')
    pairs = held.prepared(rows)
    sources = ex.read(study.ROOT / 'anchor_sources.json')
    reports = {}
    for mode in study.MODES:
        cache, _ = study.build(d, rows, pairs, mode)
        folder = study.ROOT / 'arms' / mode / 'training/median11'
        with np.load(study.ROOT / 'validation' / mode / 'predictions.npz') as cv, np.load(folder / 'predictions.npz') as delivered:
            for i, pair in enumerate(pairs):
                if 'excluded' in pair:
                    continue
                for view in ('A', 'C'):
                    key = f'{i}_{view}'
                    np.testing.assert_array_equal(cv[key], delivered[key])
                    np.testing.assert_array_equal(cv[key], cache[pair['path_' + view]])
        # Recompute from original out-of-fold predictions, without the study cache.
        pair_map = {p['path_C']: p for p in pairs if 'excluded' not in p}
        for fold, groups in enumerate(d['folds'], 1):
            subset = [r for r in rows if ex.group(r) in groups]
            parent = study.previous.BASE / f'cross_validation/fold_{fold}/matched_baseline/validation/predictions.npz'
            with np.load(parent) as old:
                for i, row in enumerate(subset):
                    if 'excluded' in row:
                        continue
                    for view in ('A', 'C'):
                        raw = pair_map[row['path_C']][view]['views'][0]
                        values = old[f'{i}_{view}'][0]
                        indices = readout.representative_indices(raw, values, mode)
                        projected = readout.apply_indices(values, indices)
                        np.testing.assert_array_equal(projected, cache[row['path_' + view]][0])
                        covered = np.zeros(len(raw), bool)
                        for start, stop in measured_candidates(raw):
                            covered[start:stop] = True
                            chosen = indices[start:stop]
                            assert np.all((chosen == -1) | ((chosen >= start) & (chosen < stop)))
                        outside = ~covered & (raw[:, 66] > .5)
                        np.testing.assert_array_equal(projected[outside], values[outside])
        reports[mode] = outputs(folder, study.ROOT / 'train_pairs.json', sources)
    protected = ex.read(study.ROOT.parent / '2026-10-28_v7_guard/declaration.json')['protected_files']
    for path, digest in protected.items():
        assert ex.sha(path) == digest
    assert not (study.ROOT.parent / 'active_model.json').exists()
    study.declare()
    receipt = dict(results=reports, prediction_replay_exact=True, observed_value_readout_verified=True,
                   dense_fallback_exact=True, protected_files_unchanged=len(protected), activation_absent=True,
                   declaration_sha256=ex.sha(study.ROOT / 'declaration.json'),
                   verifier_sha256=ex.sha(Path(__file__)), research_validation_passed=False)
    ex.write(study.ROOT / 'integrity_receipt.json', receipt)
    print('DISTRIBUTION_READOUT_VERIFIED', receipt, flush=True)


if __name__ == '__main__':
    main()
