"""Engineering-only edit-permission and rollback fixtures, separate from delivery.

No result is a ProcessingResult or ACCEPT_PREPARED package. Thresholds are
declared fixture values, not a calibrated production permission policy.
"""
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class FixturePolicy:
    version: str = 'controller-fixture-0.1'
    max_correction_cents: float = 30.
    max_disagreement_cents: float = 5.
    max_uncertainty_cents: float = 5.
    max_affected_fraction: float = .25
    min_support_seconds: float = .86

    def __post_init__(self):
        values = (self.max_correction_cents, self.max_disagreement_cents,
                  self.max_uncertainty_cents, self.max_affected_fraction, self.min_support_seconds)
        if not all(np.isfinite(v) and v > 0 for v in values) or self.max_affected_fraction > 1:
            raise ValueError('Invalid provisional fixture policy')


def waveform_hash(audio):
    return hashlib.sha256(audio.tobytes()).hexdigest()


def decision_hash(plan):
    content = {k:v for k,v in plan.items() if k != 'decision_sha256'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_source(audio, sr):
    if (not isinstance(audio, np.ndarray) or audio.dtype != np.float32 or audio.ndim != 1
            or sr != 22050 or not .25 <= len(audio)/sr <= 20.
            or not np.isfinite(audio).all() or np.max(abs(audio)) >= 1):
        raise ValueError('Fixtures require finite unclipped mono float32, 22050 Hz, 0.25-20 seconds')


def plan_fixture(audio, sr, times, corrections, alternate, uncertainty, reliable, protected,
                 policy=FixturePolicy()):
    """Withhold the entire proposed edit on any failed gate; never clip its values.

    Corrections are supplied test inputs, not inferred intended pitch. Alternate
    predictions/uncertainties in the demonstration are fixtures, not calibrated
    model uncertainty. Runtime feature/proposal production is not implemented.
    """
    check_source(audio, sr)
    times, corrections = np.asarray(times), np.asarray(corrections)
    reliable, protected = np.asarray(reliable), np.asarray(protected)
    expected = np.arange(len(audio)//441+1)*.02
    if (times.shape != expected.shape or not np.allclose(times, expected, rtol=0, atol=1e-12)
            or corrections.shape != times.shape or reliable.shape != times.shape or protected.shape != times.shape
            or reliable.dtype != bool or protected.dtype != bool or not np.isfinite(corrections).all()):
        raise ValueError('Invalid frame grid, masks or corrections')
    changed = corrections != 0
    result = dict(scope='engineering_fixture_only', policy=asdict(policy), source_sha256=waveform_hash(audio),
                  sample_count=len(audio), sample_rate_hz=sr, permitted=False, identity=not bool(changed.any()),
                  reasons=[], intervals_seconds=[], affected_fraction=0., proposal_sha256=waveform_hash(corrections),
                  requested_max_abs_cents=float(np.max(abs(corrections))), production_eligible=False)
    if not changed.any():
        result['permitted'] = True
        result['decision_sha256'] = decision_hash(result)
        return result
    if alternate is None or uncertainty is None:
        result['reasons'].append('MISSING_AGREEMENT_OR_UNCERTAINTY')
        result['decision_sha256'] = decision_hash(result)
        return result
    alternate, uncertainty = np.asarray(alternate), np.asarray(uncertainty)
    if (alternate.shape != times.shape or uncertainty.shape != times.shape
            or not np.isfinite(alternate).all() or not np.isfinite(uncertainty).all() or np.any(uncertainty < 0)):
        raise ValueError('Invalid alternate/uncertainty arrays')
    if np.any(abs(corrections[changed]) > policy.max_correction_cents):
        result['reasons'].append('CORRECTION_LIMIT')
    if np.any(abs(corrections[changed]-alternate[changed]) > policy.max_disagreement_cents):
        result['reasons'].append('BOUNDARY_DISAGREEMENT')
    if np.any(uncertainty[changed] > policy.max_uncertainty_cents):
        result['reasons'].append('UNCERTAINTY_LIMIT')
    if np.any(~reliable[changed]):
        result['reasons'].append('UNRELIABLE_VOICING')
    if np.any(protected[changed]):
        result['reasons'].append('PROTECTED_REGION')
    indices = np.flatnonzero(changed)
    spans = np.split(indices, np.flatnonzero(np.diff(indices)>1)+1)
    intervals = [(float(times[group[0]]), min(float(times[group[-1]]+.02), len(audio)/sr)) for group in spans]
    if any(end-start < policy.min_support_seconds-1e-12 for start,end in intervals):
        result['reasons'].append('INSUFFICIENT_SUPPORT')
    sample_mask = np.zeros(len(audio), dtype=bool)
    sample_times = np.arange(len(audio))/sr
    for start,end in intervals:
        sample_mask |= (sample_times >= start) & (sample_times < end)
    fraction = float(sample_mask.mean())
    result['affected_fraction'] = fraction
    if fraction > policy.max_affected_fraction:
        result['reasons'].append('AFFECTED_FRACTION_LIMIT')
    result['permitted'] = not result['reasons']
    # Failed proposals carry no usable permission intervals.
    if result['permitted']:
        result['intervals_seconds'] = intervals
    result['decision_sha256'] = decision_hash(result)
    return result


def transact_fixture(audio, sr, plan, renderer, auditor):
    """Render in a working copy and rollback failures; never publishes accepted audio.

    Renderer receives a disposable copy and the permitted intervals. Auditor
    receives separate copies of source/candidate. The preserved input cannot be
    modified by either callback. Failed candidate samples are not returned.
    """
    check_source(audio, sr)
    preserved = audio.copy()
    if (plan['scope'] != 'engineering_fixture_only' or plan['production_eligible']
            or plan.get('decision_sha256') != decision_hash(plan)
            or plan['source_sha256'] != waveform_hash(preserved) or plan['sample_rate_hz'] != sr
            or plan['sample_count'] != len(preserved)):
        raise ValueError('Plan/source mismatch or non-fixture scope')
    diagnostics = dict(scope='engineering_fixture_only', production_eligible=False, accepted_audio=None,
                       plan=copy.deepcopy(plan), outcome='WITHHELD', reasons=list(plan['reasons']), rollback=False)
    if not plan['permitted']:
        return preserved, diagnostics
    if plan['identity']:
        diagnostics['outcome'] = 'IDENTITY'
        return preserved, diagnostics
    try:
        candidate = renderer(preserved.copy(), sr, tuple(plan['intervals_seconds']))
        if not isinstance(candidate, np.ndarray) or candidate.dtype != np.float32 or candidate.shape != preserved.shape:
            raise ValueError('CANDIDATE_SAMPLE_FORMAT_OR_COUNT')
        if not np.isfinite(candidate).all() or np.max(abs(candidate)) >= 1:
            raise ValueError('CANDIDATE_NONFINITE_OR_CLIPPED')
        inside = np.zeros(len(preserved), dtype=bool)
        times = np.arange(len(preserved))/sr
        for start,end in plan['intervals_seconds']:
            inside |= (times >= start) & (times < end)
        if not np.array_equal(candidate[~inside], preserved[~inside]):
            raise ValueError('OUTSIDE_SUPPORT_CHANGED')
        measured = auditor(preserved.copy(), candidate.copy(), sr, tuple(plan['intervals_seconds']))
        diagnostics['post_render_audit'] = measured
        if not isinstance(measured, dict) or measured.get('passed') is not True:
            raise ValueError('POST_RENDER_AUDIT_FAILED')
        diagnostics.update(outcome='VERIFIED_FIXTURE_CANDIDATE', candidate_sha256=waveform_hash(candidate))
        return candidate.copy(), diagnostics
    except Exception as exc:
        diagnostics.update(outcome='ROLLED_BACK', rollback=True,
                           reasons=[str(exc)], error_type=type(exc).__name__)
        return preserved, diagnostics
