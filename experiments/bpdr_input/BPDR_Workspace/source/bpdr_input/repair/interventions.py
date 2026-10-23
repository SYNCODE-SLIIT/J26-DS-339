"""Exploratory renderer and predeclared audits; targets never enter acoustic inference."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class AuditPolicy:
    version: str = 'phase2-render-audit-0.3'
    shifts_cents: tuple = (-30., -20., -10., 10., 20., 30.)
    ramp_seconds: float = .08
    analysis_guard_seconds: float = .10
    min_plateau_seconds: float = .30
    median_error_tolerance_cents: float = 5.
    p90_error_tolerance_cents: float = 10.
    pesto_median_coarse_tolerance_cents: float = 20.
    pesto_p90_coarse_tolerance_cents: float = 40.
    tracker_agreement_cents: float = 50.
    outside_pitch_drift_tolerance_cents: float = 5.
    min_audit_frames: int = 10
    min_plateau_coverage_fraction: float = .75
    min_outside_audit_frames: int = 10
    min_outside_coverage_fraction: float = .75
    outside_p90_pitch_drift_tolerance_cents: float = 20.


def longest_span(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    return max(runs, key=lambda span: span[1]-span[0]) if runs else None


def select_support(times, reliable, policy=AuditPolicy()):
    span = longest_span(reliable)
    if span is None:
        raise ValueError('No reliable region')
    start, stop = times[span[0]], times[span[1]-1]
    # Leave analysis windows inside the agreed reliable region.
    start += policy.analysis_guard_seconds
    stop -= policy.analysis_guard_seconds
    if stop-start-2*policy.ramp_seconds-2*policy.analysis_guard_seconds < policy.min_plateau_seconds:
        raise ValueError('Reliable region too short for plateau and guarded ramps')
    return float(start), float(stop)


def envelope(times, support, ramp=.08):
    start, end = support
    if not 0 <= start < end or end-start <= 2*ramp:
        raise ValueError('Invalid local support interval')
    result = np.zeros(len(times))
    rise = (times >= start) & (times < start+ramp)
    plateau = (times >= start+ramp) & (times <= end-ramp)
    fall = (times > end-ramp) & (times < end)
    result[rise] = .5-.5*np.cos(np.pi*(times[rise]-start)/ramp)
    result[plateau] = 1.
    result[fall] = .5-.5*np.cos(np.pi*(end-times[fall])/ramp)
    return result


def render_local(audio, sr, cents, support, policy=AuditPolicy()):
    import librosa
    audio = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all() or audio.ndim != 1:
        raise ValueError('Renderer requires finite mono samples')
    if cents == 0:
        return audio.copy(), np.zeros(len(audio))
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(cents)/100, scale=False)
    if shifted.shape != audio.shape:
        raise ValueError('Pitch renderer changed sample count')
    blend = envelope(np.arange(len(audio))/sr, support, policy.ramp_seconds)
    candidate = (audio*(1-blend)+shifted*blend).astype(np.float32)
    if not np.isfinite(candidate).all() or abs(candidate).max() >= 1:
        raise ValueError('Rendered audio is non-finite or clips; no automatic normalization')
    # Samples outside support are copied bit-for-bit; crossfade pitch is not linear.
    return candidate, blend


def independent_pitch(audio, sr):
    import librosa
    # Distinct autocorrelation algorithm; not ground truth and not sufficient alone.
    return librosa.yin(audio, sr=sr, fmin=50., fmax=1500., frame_length=2048, hop_length=441)


def cents_difference(after, before, mask):
    valid = mask & np.isfinite(after) & np.isfinite(before) & (after > 0) & (before > 0)
    result = np.full(len(mask), np.nan)
    result[valid] = 1200*np.log2(after[valid]/before[valid])
    return result, valid


def audit_shift(before_features, after_features, yin_before, yin_after, reliable, support, cents, policy=AuditPolicy()):
    times = before_features['times_seconds']
    if len(yin_before) != len(times) or len(yin_after) != len(times) or not np.array_equal(times, after_features['times_seconds']):
        raise ValueError('Renderer audit has inconsistent time grids')
    start, end = support
    plateau = reliable & (times >= start+policy.ramp_seconds+policy.analysis_guard_seconds) & (times <= end-policy.ramp_seconds-policy.analysis_guard_seconds)
    outside = reliable & ((times < start-policy.analysis_guard_seconds) | (times > end+policy.analysis_guard_seconds))
    pesto_delta, pesto_valid = cents_difference(after_features['f0_hz'], before_features['f0_hz'], before_features['voiced_mask'] & after_features['voiced_mask'])
    yin_delta, yin_valid = cents_difference(yin_after, yin_before, reliable)
    common = plateau & pesto_valid & yin_valid
    def metrics(values):
        errors = abs(values[common]-cents)
        return dict(median_shift_cents=float(np.median(values[common])), median_error_cents=float(np.median(errors)), p90_error_cents=float(np.quantile(errors, .9))) if common.any() else None
    p, y = metrics(pesto_delta), metrics(yin_delta)
    outside_valid = outside & pesto_valid & yin_valid
    outside_median_error = float(max(np.median(abs(pesto_delta[outside_valid])), np.median(abs(yin_delta[outside_valid])))) if outside_valid.any() else None
    outside_p90_error = float(max(np.quantile(abs(pesto_delta[outside_valid]), .9), np.quantile(abs(yin_delta[outside_valid]), .9))) if outside_valid.any() else None
    plateau_coverage = float(common.sum()/plateau.sum()) if plateau.any() else 0.
    outside_coverage = float(outside_valid.sum()/outside.sum()) if outside.any() else 0.
    enough = int(common.sum()) >= policy.min_audit_frames and plateau_coverage >= policy.min_plateau_coverage_fraction
    fine_pass = enough and y['median_error_cents'] <= policy.median_error_tolerance_cents and y['p90_error_cents'] <= policy.p90_error_tolerance_cents and y['median_shift_cents']*cents > 0
    # Three PESTO bins/semitone can yield ~33-cent steps on peaked activations.
    # Fine accuracy is certified by the independent estimator, not quantized PESTO.
    coarse_pass = enough and p['median_error_cents'] <= policy.pesto_median_coarse_tolerance_cents and p['p90_error_cents'] <= policy.pesto_p90_coarse_tolerance_cents and p['median_shift_cents']*cents >= 0
    passed = fine_pass and coarse_pass
    passed = bool(passed and int(outside_valid.sum()) >= policy.min_outside_audit_frames
                  and outside_coverage >= policy.min_outside_coverage_fraction
                  and outside_median_error <= policy.outside_pitch_drift_tolerance_cents
                  and outside_p90_error <= policy.outside_p90_pitch_drift_tolerance_cents)
    # Only audited plateau and protected outside regions receive supervised anchors.
    anchor_mask = (common | outside_valid) if passed else np.zeros(len(times), dtype=bool)
    target = np.zeros(len(times))
    if passed:
        target[common] = -cents
    return dict(passed=passed, policy=asdict(policy), audit_frame_count=int(common.sum()),
                eligible_plateau_frame_count=int(plateau.sum()), plateau_coverage_fraction=plateau_coverage,
                pesto=p, independent_yin=y, outside_audit_frame_count=int(outside_valid.sum()),
                eligible_outside_frame_count=int(outside.sum()), outside_coverage_fraction=outside_coverage,
                outside_median_drift_cents=outside_median_error, outside_p90_drift_cents=outside_p90_error,
                target_definition='negative injected cents on audited plateau; zero outside; ramps excluded',
                fine_shift_auditor='YIN', coarse_consistency_auditor='PESTO',
                limitations='YIN fine-shift audit plus coarse PESTO consistency; human reference review remains mandatory'), anchor_mask, target


def nominal_boundaries(times, notes):
    # Timing-only annotation stream: never includes annotated pitch as model input.
    endpoints = sorted({float(edge) for note in notes['A1']
                        for edge in (note['start_seconds'], note['start_seconds']+note['duration_seconds'])})
    return np.asarray([v for v in endpoints if times[0] <= v <= times[-1]])


def boundary_track(times, endpoints):
    scores = np.zeros(len(times), dtype=np.float32)
    for endpoint in endpoints:
        scores = np.maximum(scores, np.maximum(1-abs(times-endpoint)/.04, 0))
    return scores


def corrupt_boundaries(endpoints, times, kind, seed=339, split_time=None):
    values = list(map(float, endpoints))
    rng = np.random.default_rng(seed)
    if kind == 'split':
        if len(values) < 2:
            raise ValueError('Need two boundaries for a false split')
        gaps = np.diff(values)
        index = int(np.argmax(gaps))
        inserted = (values[index]+values[index+1])/2 if split_time is None else float(split_time)
        if not times[0] <= inserted <= times[-1] or any(abs(v-inserted)<1e-6 for v in values):
            raise ValueError('False split must add a distinct boundary within the crop')
        values.append(inserted)
    elif kind == 'merge':
        if len(values) < 3:
            raise ValueError('Need an internal boundary to remove')
        values.pop(len(values)//2)
    elif kind == 'jitter':
        values = [float(np.clip(v+rng.uniform(-.06, .06), times[0], times[-1])) for v in values]
    else:
        raise ValueError('Unknown boundary corruption')
    return np.asarray(sorted(set(values)))
