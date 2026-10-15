"""Read-only diagnosis of frozen source masks; no estimator or renderer call."""
import numpy as np
from .source_estimator import nearest_indices, guarded_support

VERSION = 'frozen-support-fragmentation-0.1'


def intervals(mask, sr=22050):
    mask = np.asarray(mask)
    if mask.ndim != 1 or mask.dtype != bool or sr != 22050:
        raise ValueError('Expected boolean mono support at 22050 Hz')
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [dict(start_sample=int(lo), end_sample=int(hi), samples=int(hi-lo),
                 seconds=float((hi-lo)/sr)) for lo, hi in zip(edges[::2], edges[1::2])]


def diagnose(arrays, sr=22050):
    """Attribute ordered gate losses, without treating agreement as truth.

    Frame exclusions are exclusive in evaluation order: PESTO/confidence,
    YIN validity, then disagreement. Sample losses use all decoded samples.
    """
    t = arrays['frame_times']; f = arrays['frame_pesto_hz']
    c = arrays['frame_confidence']; y = arrays['frame_yin_hz']
    if any(np.asarray(v).shape != np.asarray(t).shape for v in (f, c, y)):
        raise ValueError('Frame grids differ')
    pesto = np.isfinite(c) & (c >= .8) & np.isfinite(f) & (f > 0)
    yin = pesto & np.isfinite(y) & (y > 0)
    reliable = yin.copy()
    reliable[yin] &= abs(1200*np.log2(y[yin]/f[yin])) <= 50.
    if not np.array_equal(reliable, arrays['frame_reliable']):
        raise ValueError('Frozen reliability differs from fixed rules')
    samples = arrays['sample_times']; n = len(samples)
    if not np.array_equal(samples, np.arange(n)/sr):
        raise ValueError('Sample time grid changed')
    mapped = reliable[nearest_indices(t, samples)]
    guarded = guarded_support(mapped, sr)
    retained = arrays['retained_support']
    if (not np.array_equal(mapped, arrays['sample_reliable']) or
            not np.array_equal(guarded, arrays['guarded_support']) or
            retained.shape != (n,) or retained.dtype != bool or np.any(retained & ~guarded)):
        raise ValueError('Frozen support masks inconsistent')
    if guarded.any():
        amp = arrays['estimated_amplitude']
        expected = guarded & (amp > .1*float(np.median(amp[guarded])))
        if not np.array_equal(expected, retained):
            raise ValueError('Frozen amplitude floor changed')
    raw = intervals(mapped, sr); gaps = intervals(~mapped, sr)
    center = float(np.median(y[reliable])) if reliable.any() else None
    outside = ((y[reliable] < .7*center) | (y[reliable] > 1.3*center)) if center else np.array([], dtype=bool)
    stages = dict(pesto_or_confidence_excluded=int((~pesto).sum()),
                  yin_invalid_after_pesto=int((pesto & ~yin).sum()),
                  disagreement_over_50c_after_validity=int((yin & ~reliable).sum()),
                  reliable=int(reliable.sum()))
    assert sum(stages.values()) == len(t)
    internal = [g for g in gaps if g['start_sample'] > 0 and g['end_sample'] < n]
    return dict(total_frames=len(t), frame_partition=stages, decoded_samples=n,
        reliable_samples=int(mapped.sum()), guard_removed_samples=int(mapped.sum()-guarded.sum()),
        guarded_samples=int(guarded.sum()), amplitude_removed_samples=int(guarded.sum()-retained.sum()),
        retained_samples=int(retained.sum()), raw_reliable_intervals=raw, internal_unreliable_intervals=internal,
        raw_runs=len(raw), raw_runs_no_longer_than_two_guards=sum(r['samples'] <= 4410 for r in raw),
        longest_raw_run_seconds=max((r['seconds'] for r in raw), default=0.),
        longest_retained_run_seconds=max((r['seconds'] for r in intervals(retained, sr)), default=0.),
        retained_runs_at_least_renderer_minimum=sum(r['samples'] >= 18963 for r in intervals(retained, sr)),
        reliable_yin_outside_global_band=int(outside.sum()),
        reliable_yin_band_denominator=int(reliable.sum()),
        meaning='Predicted support and tracker geometry only; no human voicing or accuracy inference')
