"""Separate post-seal synthetic frequency scorer; never imported by estimator."""
import numpy as np


def errors(values):
    if not len(values):return None
    if not np.isfinite(values).all():raise ValueError('Nonfinite scoring errors')
    return dict(samples=int(len(values)),median=float(np.median(values)),
                p95=float(np.quantile(values,.95)),maximum=float(np.max(values)))


def score_frequencies(global_frequency, local_frequency, retained, true_f0, true_voiced):
    if any(v.shape!=retained.shape for v in (global_frequency,local_frequency,true_f0,true_voiced)):
        raise ValueError('Synthetic scorer grids differ')
    if retained.dtype!=bool or true_voiced.dtype!=bool or not np.all(np.isfinite(true_f0)&(true_f0>0)):
        raise ValueError('Invalid synthetic scoring domain')
    domain=retained&true_voiced
    predictions=dict(B=global_frequency,C=local_frequency)
    valid={name:np.isfinite(f)&(f>0) for name,f in predictions.items()}
    common=domain&valid['B']&valid['C']
    result=dict(retained_denominator=int(retained.sum()),retained_true_voiced_denominator=int(domain.sum()),
                retained_silent_samples=int((retained&~true_voiced).sum()),common_valid_voiced_samples=int(common.sum()),arms={})
    for name,f in predictions.items():
        scored=domain&valid[name]
        result['arms'][name]=dict(valid_on_retained_samples=int((retained&valid[name]).sum()),
            invalid_on_retained_samples=int((retained&~valid[name]).sum()),
            invalid_on_retained_voiced_samples=int((domain&~valid[name]).sum()),
            scored_voiced_samples=int(scored.sum()),
            absolute_error_cents=errors(abs(1200*np.log2(f[scored]/true_f0[scored]))),
            common_valid_absolute_error_cents=errors(abs(1200*np.log2(f[common]/true_f0[common]))))
    return result


def score_saved_source(profile, audio, sr, frozen, local):
    # This import/construction is confined to scoring after all predictions seal.
    from bpdr_input.experiments.audio.framewise_fixtures import constructed, signal
    _,f0,amplitude=constructed(profile,sr)
    if not np.array_equal(audio,signal(f0,amplitude,sr)):
        raise ValueError('Frozen source does not match synthetic construction')
    return score_frequencies(frozen['estimated_frequency_hz'],local['local_frequency_hz'],
                             frozen['retained_support'],f0,amplitude>0)
