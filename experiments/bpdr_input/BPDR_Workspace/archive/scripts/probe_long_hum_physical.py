"""Bounded physical construction probe for a 0.24-second repair after 10 seconds."""

from io import BytesIO
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

import calibrate_short_support_audit as calibration
import prepare_humtrans_short_training as source_study
from bpdr_input.repair.held_voice_policy import spans


ROOT = source_study.ROOT.parent / '2026-11-06_long_hum_physical_probe'
read, write, sha = source_study.read, source_study.write, source_study.sha
WIDTH = .24
SHIFTS = (-20., 20.)
RATE = 22050


def pcm24(audio):
    stream = BytesIO()
    sf.write(stream, audio, RATE, format='WAV', subtype='PCM_24')
    stream.seek(0)
    decoded, rate = sf.read(stream, dtype='float32')
    assert rate == RATE and decoded.shape == audio.shape
    return decoded


def working_audio(path):
    native, rate = sf.read(path, dtype='float32', always_2d=True)
    channel = int(np.argmax(np.mean(native.astype(np.float64) ** 2, axis=0)))
    audio = native[:15 * rate, channel]
    if rate != RATE:
        divisor = gcd(rate, RATE)
        audio = resample_poly(audio, RATE // divisor, rate // divisor).astype(np.float32)
    assert 10 < len(audio) / RATE <= 15
    return audio


def tracking(audio, times):
    return (calibration.coarse_features(audio, RATE, source_study.WS / 'checkpoints'),
            calibration.fine_hz(audio, RATE, times),
            calibration.short.cross_correlation_pitch(audio, RATE, times))


def probe(audio, geometry, source_id):
    if not np.isfinite(audio).all() or abs(audio).max() >= 1:
        assert geometry['excluded'] == 'INVALID_SOURCE_SAMPLES'
        return [dict(source_id=source_id, cents=cents, excluded=geometry['excluded']) for cents in SHIFTS]
    audio = pcm24(audio)
    analysis = calibration.recovery.measured(audio)
    times = analysis['features']['times_seconds']
    reliable = analysis['reliable']
    region = max(spans(reliable & (times > 10)), key=lambda span: span[1] - span[0], default=(0, 0))
    assert region[1] - region[0] == geometry['after_10']['longest_frames']
    if WIDTH not in geometry['after_10']['widths']:
        return [dict(source_id=source_id, cents=cents, excluded='INSUFFICIENT_RELIABLE_CONTEXT') for cents in SHIFTS]
    center = float(np.mean(region) * .02)
    start, end = center - WIDTH / 2, center + WIDTH / 2
    assert start - 10 >= .15 and end <= len(audio) / RATE
    before, fine_before, cc_before = tracking(audio, times)
    sample_times = np.arange(len(audio)) / RATE
    outside = (sample_times <= start) | (sample_times >= end)
    results = []
    for cents in SHIFTS:
        row = dict(source_id=source_id, cents=cents, support_seconds=[start, end])
        curve = np.where((times >= start) & (times < end), cents, 0.)
        try:
            candidate, _ = calibration.render_candidate(audio, RATE, times, curve, [(start, end)], calibration.RENDER)
        except ValueError as exc:
            results.append(dict(row, excluded='RENDER_REFUSED', reason=str(exc)))
            continue
        decoded = pcm24(candidate)
        np.testing.assert_array_equal(decoded[outside], audio[outside])
        after, fine_after, cc_after = tracking(decoded, times)

        def audit(features, fine, cc, support, requested):
            return calibration.short.audit_short_curve(before, features, fine_before, fine, cc_before, cc,
                                                       reliable, [support], requested)

        actual = audit(after, fine_after, cc_after, (start, end), curve)
        offset = 1. if end + 1. <= len(audio) / RATE else -1.
        displaced = (start + offset, end + offset)
        misplaced = np.where((times >= displaced[0]) & (times < displaced[1]), cents, 0.)
        controls = dict(
            unchanged=audit(before, fine_before, cc_before, (start, end), curve),
            opposite=audit(after, fine_after, cc_after, (start, end), -curve),
            wrong_magnitude=audit(after, fine_after, cc_after, (start, end), curve / 3),
            wrong_location=audit(after, fine_after, cc_after, displaced, misplaced),
        )
        row.update(physical_audit=actual, controls=controls, exact_outside=True)
        if not actual['passed']:
            results.append(dict(row, excluded='THREE_TRACKER_PHYSICAL_AUDIT_FAILED'))
            continue
        altered = calibration.recovery.measured(decoded)['reliable']
        common = reliable & altered & before['voiced_mask'] & after['voiced_mask']
        common &= np.isfinite(fine_before) & np.isfinite(fine_after) & (fine_before > 0) & (fine_after > 0)
        plateau = (times >= start + .09) & (times <= end - .09) & common
        plateau &= np.isfinite(cc_before) & np.isfinite(cc_after) & (cc_before > 0) & (cc_after > 0)
        left = (times >= region[0] * .02) & (times < start - .07) & altered
        right = (times > end + .07) & (times < region[1] * .02) & altered
        row.update(plateau_frames=int(plateau.sum()), left_frames=int(left.sum()), right_frames=int(right.sum()))
        if min(row['plateau_frames'], row['left_frames'], row['right_frames']) < 3:
            row['excluded'] = 'INSUFFICIENT_RETAINED_PLATEAU_OR_CONTEXT'
        results.append(row)
    return results


def main():
    torch.set_num_threads(2)
    source_study.check_lock()
    geometry = ROOT.parent / 'long_hum_geometry_preflight_2026-11-06.json'
    source_manifest = source_study.ROOT / 'declaration.json'
    cases = read(geometry)['cases']
    assert read(geometry)['parent_declaration_sha256'] == sha(source_manifest) and len(cases) == 7
    sources = {row['key']: row for row in read(source_manifest)['sources']}
    declaration = dict(geometry_sha256=sha(geometry), source_manifest_sha256=sha(source_manifest),
                       runner_sha256=sha(Path(__file__)), width_seconds=WIDTH, shifts=list(SHIFTS),
                       attempts=14, production_eligible=False)
    ROOT.mkdir(exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == declaration
    else:
        write(path, declaration)
    rows = []
    for case in cases:
        source = sources[case['source_id']]
        assert source['split'] == 'train' and sha(source['path']) == source['sha256']
        observed = probe(working_audio(source['path']), case, source['key'])
        rows.extend(observed)
        print('LONG_HUM_PHYSICAL_SOURCE', source['key'],
              [row.get('excluded', 'SCOREABLE') for row in observed], flush=True)
    assert len(rows) == 14
    false_accepts = sum(control['passed'] for row in rows for control in row.get('controls', {}).values())
    result = dict(declaration_sha256=sha(path), rows=rows, negative_control_accepts=false_accepts,
                  scoreable=sum('excluded' not in row for row in rows), model_fits=0)
    result = calibration.norm(result)
    output = ROOT / 'results.json'
    if output.exists():
        assert read(output) == result
        print('LONG_HUM_PHYSICAL_REPLAY_MATCH', flush=True)
    else:
        write(output, result)
    assert false_accepts == 0, 'Negative control accepted'


if __name__ == '__main__':
    main()
