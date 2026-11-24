"""Independent source/preprocessing, short-audit, feature and label replay."""
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

import prepare_humtrans_short_training as run
from verify_short_phrase_training import check_arrays


def compare_arrays(path, expected):
    with np.load(path) as saved:
        assert set(saved.files) == set(expected)
        for key, value in expected.items():
            np.testing.assert_array_equal(saved[key], value, err_msg=str(path) + ' ' + key)
    return len(expected)


def main():
    torch.set_num_threads(2)
    run.check_lock()
    root = run.ROOT
    d = run.read(root / 'declaration.json')
    assert d['reconciliation_sha256'] == run.sha(run.RECONCILIATION)
    for path, expected in d['source_hashes'].items():
        assert run.sha(path) == expected
    for path, expected in run.read(root / 'input_hashes.json').items():
        assert run.sha(path) == expected
    rows = run.read(root / 'train_pairs.json')
    originals = run.read(root / 'train_originals.json')
    assert len(rows) == 192 and len(originals) == 12
    recreated = root / 'verification_recreated'
    recreated.mkdir(exist_ok=True)
    cache = {}; arrays = 0; decoded_sources = 0; audits = 0; controls = 0; labels = 0; exclusions = 0
    for original in originals:
        source = original['source']
        assert source['split'] == 'train' and source['person_id'] in {'F01', 'F02', 'F03', 'F04', 'F05', 'M01'}
        assert run.sha(source['path']) == source['sha256']
        native, sr = sf.read(source['path'], dtype='float32', always_2d=True)
        channel = int(np.argmax(np.mean(native.astype(np.float64) ** 2, axis=0)))
        y = native[:min(len(native), 10 * sr), channel]
        divisor = gcd(sr, 22050)
        if sr != 22050:
            y = resample_poly(y, 22050 // divisor, sr // divisor).astype(np.float32)
        assert channel == original['source_channel'] and len(y) / 22050 == original['duration_seconds']
        if len(y) < 5 * 22050 or not np.isfinite(y).all() or abs(y).max() >= 1:
            assert original['excluded'] == 'INVALID_SOURCE_DURATION_OR_SAMPLES'
            continue
        temp = recreated / (original['source_id'] + '.wav')
        sf.write(temp, y, 22050, subtype='PCM_24')
        assert run.sha(temp) == run.sha(original['path_A'])
        y, rate = sf.read(temp, dtype='float32'); assert rate == 22050
        a = run.calibration.recovery.measured(y)
        arrays += check_arrays(original['analysis_A'], a)
        times = a['features']['times_seconds']
        regions = run.segment_notes(a['views'][0][:, 64], a['reliable'], run.RulePolicy(note_split_cents=70.))
        region = max(regions, key=lambda r: r[1] - r[0]) if regions else None
        assert run.norm(region) == original['longest_region_frames']
        bf = run.calibration.coarse_features(y, 22050, run.WS / 'checkpoints')
        yb = run.calibration.fine_hz(y, 22050, times)
        cb = run.calibration.short.cross_correlation_pitch(y, 22050, times)
        arrays += compare_arrays(root / 'training' / original['source_id'] / 'before_measurements.npz',
                                 dict(**bf, yin=yb, cc=cb, reliable=a['reliable']))
        cache[original['source_id']] = (y, a, region, bf, yb, cb)
        decoded_sources += 1
    for row in rows:
        if row['source_id'] not in cache:
            assert row['excluded'] == 'INVALID_SOURCE_DURATION_OR_SAMPLES'; exclusions += 1; continue
        y, a, region, bf, yb, cb = cache[row['source_id']]
        width = row['width_seconds']
        if region is None or (region[1] - region[0]) * .02 < width + .30 - 1e-12:
            assert row['excluded'] == 'NO_SINGLE_RELIABLE_REGION_WITH_150MS_CONTEXT'; exclusions += 1; continue
        times = a['features']['times_seconds']
        center = float(np.mean(region) * .02)
        start, end = center - width / 2, center + width / 2
        assert row['support_seconds'] == [start, end]
        c_audio, sr = sf.read(row['path_C'], dtype='float32')
        assert sr == 22050 and y.shape == c_audio.shape and np.isfinite(c_audio).all() and abs(c_audio).max() < 1
        st = np.arange(len(y)) / sr
        outside = (st <= start) | (st >= end)
        np.testing.assert_array_equal(c_audio[outside], y[outside])
        af = run.calibration.coarse_features(c_audio, sr, run.WS / 'checkpoints')
        ya = run.calibration.fine_hz(c_audio, sr, times)
        ca = run.calibration.short.cross_correlation_pitch(c_audio, sr, times)
        curve = np.where((times >= start) & (times < end), row['cents'], 0.)
        audit = run.calibration.short.audit_short_curve(bf, af, yb, ya, cb, ca, a['reliable'], [(start, end)], curve)
        saved = run.read(row['audit_path'])
        assert run.norm(audit) == row['audit'] == saved['audit']; audits += 1
        for name in ('unchanged', 'opposite'):
            opposite = name == 'opposite'
            control = run.calibration.short.audit_short_curve(bf, af if opposite else bf, yb, ya if opposite else yb,
                cb, ca if opposite else cb, a['reliable'], [(start, end)], -curve if opposite else curve)
            assert run.norm(control) == saved['controls'][name] and not control['passed']; controls += 1
        arrays += compare_arrays(Path(row['path_C']).parent / 'measurements.npz', dict(**af, yin=ya, cc=ca, requested=curve))
        if not audit['passed']:
            assert row['excluded'] == 'THREE_TRACKER_PHYSICAL_AUDIT_FAILED'; continue
        c = run.calibration.recovery.measured(c_audio)
        reliable = a['reliable'] & c['reliable'] & bf['voiced_mask'] & af['voiced_mask'] & np.isfinite(yb) & np.isfinite(ya) & (yb > 0) & (ya > 0)
        plateau = (times >= start + .09) & (times <= end - .09) & reliable & np.isfinite(cb) & np.isfinite(ca) & (cb > 0) & (ca > 0)
        zero = ((times < start - .07) | (times > end + .07)) & reliable
        left = (times >= region[0] * .02) & (times < start - .07) & c['reliable']
        right = (times > end + .07) & (times < region[1] * .02) & c['reliable']
        if plateau.sum() < 3 or min(left.sum(), right.sum()) < 3:
            assert row['excluded'] == 'INSUFFICIENT_RETAINED_PLATEAU_OR_CONTEXT'; continue
        assert 'excluded' not in row
        arrays += compare_arrays(row['analysis_C'], dict(raw_views=np.stack(c['views']), reliable=c['reliable'], fine=c['fine'], times_seconds=times))
        arrays += compare_arrays(row['target_path'], dict(target=np.where(plateau, -row['cents'], 0.).astype(np.float32),
            mask=plateau | zero, plateau=plateau, known_zero=zero, times=times))
        labels += 1
    summary = run.read(root / 'summary.json')
    assert labels == summary['valid']
    run.write(root / 'integrity_receipt.json', dict(source_records_checked=12, originals_recreated=decoded_sources,
        planned_attempts=192, construction_exclusions_replayed=exclusions, decoded_audits_replayed=audits,
        negative_controls_replayed=controls, arrays_reinferred=arrays, label_files_verified=labels,
        declaration_sha256=run.sha(root / 'declaration.json'), train_pairs_sha256=run.sha(root / 'train_pairs.json'),
        train_originals_sha256=run.sha(root / 'train_originals.json'), input_hashes_sha256=run.sha(root / 'input_hashes.json'),
        summary_sha256=run.sha(root / 'summary.json'), verifier_source_sha256=run.sha(__file__),
        development_audio_accessed=False, unseen_audio_accessed=False, guide_notes_loaded=False,
        research_training_eligible=False, production_eligible=False))
    print('HUMTRANS_SHORT_DATA_INDEPENDENT_REPLAY_PASS', labels, audits, controls, arrays, flush=True)


if __name__ == '__main__':
    main()
