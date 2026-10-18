"""Replay existing detuning artifacts and diagnose failures without changing any gate."""
from bpdr_input import COMPONENT_ROOT, module_source
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.audio.analysis import extract
from bpdr_input.preparation.sources.data_builder import validate_run, write_json
from bpdr_input.repair.interventions import AuditPolicy, audit_shift, independent_pitch, envelope, cents_difference
from bpdr_input.audio.pipeline import checksum


def failure_reasons(m, p=AuditPolicy()):
    reasons = []
    checks = [('PLATEAU_FRAME_COUNT', m['audit_frame_count'] >= p.min_audit_frames),
              ('PLATEAU_COVERAGE', m['plateau_coverage_fraction'] >= p.min_plateau_coverage_fraction),
              ('OUTSIDE_FRAME_COUNT', m['outside_audit_frame_count'] >= p.min_outside_audit_frames),
              ('OUTSIDE_COVERAGE', m['outside_coverage_fraction'] >= p.min_outside_coverage_fraction)]
    for name, passed in checks:
        if not passed:
            reasons.append(name)
    for key, label, median_limit, p90_limit in [('independent_yin', 'FINE', p.median_error_tolerance_cents, p.p90_error_tolerance_cents),
                                               ('pesto', 'COARSE', p.pesto_median_coarse_tolerance_cents, p.pesto_p90_coarse_tolerance_cents)]:
        values = m[key]
        if values is None:
            reasons.append(label+'_MISSING')
            continue
        if values['median_error_cents'] > median_limit:
            reasons.append(label+'_MEDIAN_ERROR')
        if values['p90_error_cents'] > p90_limit:
            reasons.append(label+'_P90_ERROR')
        sign = values['median_shift_cents']*m['requested_cents']
        if (label == 'FINE' and sign <= 0) or (label == 'COARSE' and sign < 0):
            reasons.append(label+'_DIRECTION')
    for key, label, limit in [('outside_median_drift_cents', 'OUTSIDE_MEDIAN_DRIFT', p.outside_pitch_drift_tolerance_cents),
                              ('outside_p90_drift_cents', 'OUTSIDE_P90_DRIFT', p.outside_p90_pitch_drift_tolerance_cents)]:
        if m[key] is None or m[key] > limit:
            reasons.append(label)
    for key in ('identity_pass', 'same_sample_count', 'outside_support_exact'):
        if not m[key]:
            reasons.append(key.upper())
    return reasons


def pitch_metrics(before, after, mask, cents):
    delta, valid = cents_difference(after, before, mask)
    if not valid.any():
        return dict(frames=0, median_error_cents=None, p90_error_cents=None)
    errors = abs(delta[valid]-cents)
    return dict(frames=int(valid.sum()), median_error_cents=float(np.median(errors)),
                p90_error_cents=float(np.quantile(errors, .9)))


def audit_region(times, reliable, support, p=AuditPolicy()):
    start, stop = support
    return reliable & (times >= start+p.ramp_seconds+p.analysis_guard_seconds) & (times <= stop-p.ramp_seconds-p.analysis_guard_seconds)


def window_comparison(local, full, times, plateau, sr, support, p=AuditPolicy()):
    """Check complete centered YIN windows, not just their center samples."""
    blend = envelope(np.arange(len(local))/sr, support, p.ramp_seconds)
    differences, inside = [], []
    for t in times[plateau]:
        center = int(round(t*sr))
        lo, hi = center-1024, center+1024
        if lo < 0 or hi > len(local):
            inside.append(False)
            continue
        inside.append(bool(np.all(blend[lo:hi] == 1)))
        differences.append(float(np.max(abs(local[lo:hi]-full[lo:hi]))))
    return dict(yin_windows_fully_in_unit_blend=bool(inside and all(inside)),
                max_window_sample_difference=max(differences) if differences else None,
                window_samples=2048, scope='YIN windows only; PESTO context is assessed separately')


def controls(p):
    """Analytically shifted harmonic signals separate estimator error from renderer error."""
    import librosa
    sr = 22050
    times = np.arange(4*sr)/sr
    rows = []
    for name, f0 in [('constant_440', np.full(len(times), 440.)),
                     ('vibrato_220', 220.*2**(35*np.sin(2*np.pi*5*times)/1200))]:
        def signal(cents):
            phase = 2*np.pi*np.cumsum(f0*2**(cents/1200))/sr
            return (.12*(np.sin(phase)+.4*np.sin(2*phase)+.2*np.sin(3*phase))).astype(np.float32)
        original = signal(0.)
        baseline = independent_pitch(original, sr)
        frame_times = np.arange(len(baseline))*.02
        mask = (frame_times >= 1.) & (frame_times <= 3.)
        for cents in p.shifts_cents:
            oracle = signal(cents)
            rendered = librosa.effects.pitch_shift(original, sr=sr, n_steps=cents/100, scale=False)
            rows.append(dict(profile=name, cents=cents,
                             analytic_oracle=pitch_metrics(baseline, independent_pitch(oracle, sr), mask, cents),
                             phase_vocoder=pitch_metrics(baseline, independent_pitch(rendered, sr), mask, cents),
                             meaning='Constructed signals only; not evidence of real-vocal accuracy or audible quality'))
    return rows


def run(evidence_path):
    import librosa
    import torch
    evidence_path = Path(evidence_path).resolve()
    component = COMPONENT_ROOT
    if not evidence_path.is_relative_to(component):
        raise ValueError('Evidence must be inside BPDR')
    prior = json.loads(evidence_path.read_text())
    p = AuditPolicy()
    if prior['run_status'] != 'complete' or prior['policy'] != json.loads(json.dumps(asdict(p))):
        raise ValueError('Completed active-policy evidence required')
    prepared = Path(prior['preparation_path'])
    validate_run(prepared)
    records = {r['record_id']: r for r in json.loads((prepared/'manifest.json').read_text())['records']}
    root = component/'runs/audit_diagnostics'/uuid.uuid4().hex
    root.mkdir(parents=True)
    modules = ('audit_diagnostics', 'analysis', 'interventions')
    snapshots = root/'source_code_at_run'
    snapshots.mkdir()
    code = {name: checksum(module_source(name+'.py')) for name in modules}
    for name in modules:
        (snapshots/(name+'.py')).write_bytes((module_source(name+'.py')).read_bytes())
    result = dict(run_status='running', data_status='exploratory_not_research', training_ready_examples=0,
                  source_evidence_path=str(evidence_path), source_evidence_sha256=checksum(evidence_path),
                  policy=asdict(p), code_sha256=code, librosa_version=librosa.__version__,
                  librosa_effects_sha256=checksum(Path(librosa.effects.__file__)), torch_version=torch.__version__,
                  cpu_threads=torch.get_num_threads(), comparisons=[], synthetic_controls=[],
                  limitations=['YIN/PESTO estimates are not real-vocal ground truth', 'No thresholds, renderer or eligibility were changed',
                               'Full-shift comparison is diagnostic; it changes protected outside audio and is not an admissible local renderer'])
    write_json(root/'diagnostics.json', result)
    print('DIAGNOSTIC_RUN:', root, flush=True)
    try:
        for probe in prior['probes']:
            if not probe['shifts']:
                continue
            record = records[probe['record_id']]
            if record['split'] != 'train' or record['training_eligible']:
                raise ValueError('Only exploratory training material allowed')
            if checksum(record['accepted_audio_path']) != probe['accepted_sha256']:
                raise ValueError('Original accepted audio changed')
            audio, sr = sf.read(record['accepted_audio_path'], dtype='float32')
            before, analyzer = extract(audio, sr, component/'checkpoints')
            with np.load(record['screen_masks']['path']) as archive:
                reliable, times = archive['reliable_mask'], archive['times_seconds']
            if not np.array_equal(times, before['times_seconds']):
                raise ValueError('Diagnostic time grids differ')
            yin_before = independent_pitch(audio, sr)
            support = probe['geometry']['support_seconds']
            plateau = audit_region(times, reliable, support, p)
            for shift, saved in probe['shifts'].items():
                print('REPLAY:', probe['record_id'], shift, flush=True)
                artifact = saved['artifact']
                if checksum(artifact['path']) != artifact['sha256']:
                    raise ValueError('Saved detuned artifact changed')
                local, shifted_sr = sf.read(artifact['path'], dtype='float32')
                if shifted_sr != sr or len(local) != len(audio):
                    raise ValueError('Invalid saved artifact format')
                cents = float(shift)
                after, _ = extract(local, sr, component/'checkpoints')
                yin_after = independent_pitch(local, sr)
                measured, _, _ = audit_shift(before, after, yin_before, yin_after, reliable, support, cents, p)
                measured.update(requested_cents=cents, identity_pass=probe['identity_pass'],
                                same_sample_count=True, outside_support_exact=bool(np.array_equal(audio[envelope(np.arange(len(audio))/sr, support, p.ramp_seconds) == 0],
                                                                                                 local[envelope(np.arange(len(audio))/sr, support, p.ramp_seconds) == 0])))
                reasons = failure_reasons(measured, p)
                if bool(not reasons) != measured['passed']:
                    raise ValueError('Failure classification disagrees with active audit')
                if measured['passed'] != saved['passed']:
                    raise ValueError('Replay differs from saved audit; preserve diagnostic failure')
                full = librosa.effects.pitch_shift(audio, sr=sr, n_steps=cents/100, scale=False)
                full_features, _ = extract(full, sr, component/'checkpoints')
                full_yin = independent_pitch(full, sr)
                common = plateau & before['voiced_mask'] & after['voiced_mask'] & np.isfinite(yin_after) & (yin_after > 0)
                lost = plateau & ~common
                entry = dict(record_id=record['record_id'], source_id=record['source_id'], group_id=record['group_id'],
                             cents=cents, passed=measured['passed'], saved_pass=saved['passed'], audit=measured,
                             reasons=reasons, analyzer=analyzer,
                             yin_all_original_plateau=pitch_metrics(yin_before, yin_after, plateau, cents),
                             yin_pesto_rejected_frames=pitch_metrics(yin_before, yin_after, lost, cents),
                             plateau_frames=int(plateau.sum()), lost_common_frames=int(lost.sum()),
                             after_pesto_unvoiced_frames=int((plateau & ~after['voiced_mask']).sum()),
                             lost_after_confidence_median=float(np.median(after['confidence'][lost])) if lost.any() else None,
                             full_shift=window_comparison(local, full, times, plateau, sr, support, p),
                             full_shift_yin=pitch_metrics(yin_before, full_yin, plateau, cents),
                             full_shift_pesto_voiced_fraction=float(full_features['voiced_mask'][plateau].mean()))
                entry['full_shift']['pesto_voicing_mask_equal_on_plateau'] = bool(np.array_equal(after['voiced_mask'][plateau], full_features['voiced_mask'][plateau]))
                result['comparisons'].append(entry)
                write_json(root/'diagnostics.json', result)
        result['synthetic_controls'] = controls(p)
        counts = Counter(reason for r in result['comparisons'] for reason in r['reasons'])
        result['summary'] = dict(replayed_variants=len(result['comparisons']), passing_variants=sum(r['passed'] for r in result['comparisons']),
                               replay_agreement=True, failure_reason_counts=dict(counts),
                               failure_counts_overlap=True, training_ready_examples=0)
        validate_run(prepared)
        if checksum(evidence_path) != result['source_evidence_sha256'] or any(checksum(module_source(name+'.py')) != digest for name,digest in code.items()):
            raise ValueError('Inputs/code changed during diagnosis')
        result['run_status'] = 'complete'
        write_json(root/'diagnostics.json', result)
        print(json.dumps(result['summary'], indent=2), flush=True)
    except BaseException as exc:
        result.update(run_status='failed', failure=dict(type=type(exc).__name__, reason=str(exc)))
        write_json(root/'diagnostics.json', result)
        raise
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    run(args.evidence)


if __name__ == '__main__':
    main()
