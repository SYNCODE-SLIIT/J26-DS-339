"""Independent replay of cached acoustic tracks and fixed-reference measurements."""
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import evaluate_fcpe_measurement as study


def main():
    torch.set_num_threads(2)
    ex = study.reference.ex
    tracker = study.measurement.FCPEMeasurement(study.DEPENDENCIES)
    d = study.declare(tracker)
    data = ex.read(study.ROOT / 'real_measurements.json')
    assert data['declaration_sha256'] == ex.sha(study.ROOT / 'declaration.json')
    assert data['diagnostic_only'] and not data['selection_eligible'] and not data['model_activated']
    sources = ex.read(study.ROOT / 'anchor_sources.json')
    assert len(sources) == len(data['originals']) == 50
    receipts = sorted((study.ROOT / 'tracks').glob('*.json'))
    replayed, max_error, tracks = 0, 0., {}
    for receipt in receipts:
        saved = ex.read(receipt); array = receipt.with_suffix('.npz')
        assert saved['source_sha256'] == ex.sha(saved['source'])
        assert saved['array_sha256'] == ex.sha(array)
        assert saved['declaration_sha256'] == ex.sha(study.ROOT / 'declaration.json')
        audio, rate = sf.read(saved['source'], dtype='float32')
        times, f0, metadata = tracker.native(audio, rate)
        assert metadata == saved['metadata']
        with np.load(array) as z:
            np.testing.assert_array_equal(times, z['times'])
            np.testing.assert_allclose(f0, z['f0'], atol=1e-4, rtol=1e-7)
            max_error = max(max_error, float(abs(f0 - z['f0']).max()))
            tracks[saved['source_sha256']] = (z['times'].copy(), z['f0'].copy())
        replayed += 1
        if replayed % 25 == 0:
            print('FCPE_TRACKS_REPLAYED', replayed, flush=True)
    assert replayed == 158, 'Expected50 originals plus108 altered recordings'
    for source, measured in zip(sources, data['originals'], strict=True):
        assert measured['source_sha256'] == source['source']['sha256']
        times, f0 = tracks[ex.sha(source['working_audio'])]
        assert measured['native_frames'] == len(f0)
        assert measured['detected_frames'] == int((f0 > 0).sum())
    rows = ex.read(study.ROOT / 'train_pairs.json')
    pairs = study.reference.training.held.prepared(rows)
    recomputed = {k: [] for k in ('yin', 'pesto', 'fcpe')}
    for index, pair in enumerate(pairs):
        if 'excluded' in pair:
            for name in recomputed:
                assert data['results'][name]['cases'][index]['excluded'] == pair['excluded']
                recomputed[name].append(dict(index=index, group=pair['group'], excluded=pair['excluded']))
            continue
        grid = pair['C']['features']['times_seconds']; fcpe = {}
        for view in ('A', 'C'):
            native_times, native_f0 = tracks[ex.sha(pair['path_' + view])]
            aligned, valid, indices = study.measurement.align_native(native_times, native_f0, grid)
            assert np.isnan(aligned[~valid]).all()
            assert np.all(abs(grid[valid] - native_times[indices[valid]]) <= .005 + 1e-12)
            assert np.all((native_f0[indices[valid]] >= 50.) & (native_f0[indices[valid]] <= 1500.))
            fcpe[view] = study.hz_cents(aligned)
        conditions = dict(fcpe=fcpe,
            yin={v: np.where(pair[v]['reliable'], pair[v]['fine'], np.nan) for v in ('A', 'C')},
            pesto={v: np.where(pair[v]['features']['voiced_mask'], study.hz_cents(pair[v]['features']['f0_hz']), np.nan) for v in ('A', 'C')})
        plateau = pair['mask'] & (pair['target'] != 0)
        lo, hi = pair['support_seconds']
        outside = pair['mask'] & (pair['target'] == 0) & ((grid < lo - .1) | (grid > hi + .1))
        for name, values in conditions.items():
            actual = data['results'][name]['cases'][index]
            assert actual['index'] == index and actual['group'] == pair['group']
            delta = values['C'] - values['A']
            for region, mask, wanted in (('plateau', plateau, pair['cents']), ('outside', outside, 0.)):
                valid = mask & np.isfinite(delta); count = int(mask.sum())
                error = abs(delta[valid] - wanted)
                stats = actual[region]
                assert stats['reference_frames'] == count and stats['observed_frames'] == int(valid.sum())
                assert stats['missing_frames'] == int(np.sum(mask & ~np.isfinite(delta)))
                if valid.any():
                    np.testing.assert_allclose(stats['conditional_mae_cents'], error.mean(), atol=1e-9)
                if count:
                    np.testing.assert_allclose(stats['within5_over_reference_percent'], 100 * np.sum(error <= 5.) / count, atol=1e-9)
                    np.testing.assert_allclose(stats['within10_over_reference_percent'], 100 * np.sum(error <= 10.) / count, atol=1e-9)
            recomputed[name].append(actual)
    for name, cases in recomputed.items():
        aggregate = study.macro(cases)
        assert aggregate['summary'] == data['results'][name]['summary']
        assert aggregate['by_singer'] == data['results'][name]['by_singer']
        assert aggregate['summary']['attempted'] == 200
        assert aggregate['summary']['reference_scoreable'] == 108 and aggregate['summary']['excluded'] == 92
    protected = ex.read(study.ROOT.parent / '2026-10-28_v7_guard/declaration.json')['protected_files']
    for path, digest in protected.items():
        assert ex.sha(path) == digest
    assert not (study.ROOT.parent / 'active_model.json').exists()
    study.declare(tracker)
    receipt = dict(recording_tracks_replayed=replayed, maximum_replay_error_hz=max_error,
                   reference_cases_verified=108, attempts_retained=200, exclusions_retained=92,
                   native_time_mapping_and_missingness_verified=True, protected_files_unchanged=len(protected),
                   declaration_sha256=ex.sha(study.ROOT / 'declaration.json'),
                   measurements_sha256=ex.sha(study.ROOT / 'real_measurements.json'),
                   verifier_sha256=ex.sha(Path(__file__)), selection_eligible=False,
                   model_activated=False, research_validation_passed=False)
    ex.write(study.ROOT / 'integrity_receipt.json', receipt)
    print('FCPE_MEASUREMENTS_VERIFIED', receipt, flush=True)


if __name__ == '__main__':
    main()
