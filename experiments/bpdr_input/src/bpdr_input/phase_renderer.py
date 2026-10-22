"""Experimental ramp-only analytic amplitude/phase joining; no production permission."""
import numpy as np
from scipy.signal import hilbert
from .framewise_renderer import RendererPolicy,render_candidate
from .framewise_diagnostics import private_resynthesis
from .interventions import envelope

VERSION='praat-polar-join-candidate-0.1'
AMPLITUDE_FLOOR=1e-8 # Numerical refusal floor only; not calibrated voicing/uncertainty.


def polar_ramp(source, full, weight):
    """Continuous relative-phase branch anchored near zero at the half-blend sample.

    These are full-band analytic samples, not a fundamental or intended-note oracle.
    Interpolating phase can itself alter instantaneous frequency; audit afterward.
    """
    source=np.asarray(source);full=np.asarray(full);weight=np.asarray(weight)
    if (source.ndim!=1 or not len(source) or source.shape!=full.shape or source.shape!=weight.shape
            or any(not np.isfinite(v).all() for v in (source,full,weight))
            or np.any((weight<0)|(weight>1))):raise ValueError('Invalid polar ramp arrays')
    if np.any(abs(source)<=AMPLITUDE_FLOOR) or np.any(abs(full)<=AMPLITUDE_FLOOR):
        raise ValueError('Unstable analytic phase near zero amplitude; no fallback join')
    gap=np.unwrap(np.angle(full*np.conj(source)))
    midpoint=int(np.argmin(abs(weight-.5)))
    gap-=2*np.pi*np.floor((gap[midpoint]+np.pi)/(2*np.pi))
    amplitude=(1-weight)*abs(source)+weight*abs(full)
    result=amplitude*np.exp(1j*(np.angle(source)+weight*gap))
    return result,gap


def render_phase_candidate(audio,sr,times,correction_cents,intervals,policy=RendererPolicy()):
    """Reuse original request/PSOLA checks, then alter only the two 80-ms ramps."""
    intervals=tuple(intervals)
    baseline,diagnostic=render_candidate(audio,sr,times,correction_cents,intervals,policy)
    diagnostic.update(candidate_version=VERSION,production_eligible=False,accepted_audio=None,
        join_definition='Linear analytic amplitude, ramp-local continuous relative-phase interpolation; branch anchored at half blend',
        phase_amplitude_floor=AMPLITUDE_FLOOR,fundamental_or_intended_note_input=False,
        phase_join_is_not_a_linear_pitch_ramp=True)
    if diagnostic['identity']:return baseline,diagnostic,{}
    full=private_resynthesis(audio,sr,times,correction_cents,policy)
    st=np.arange(len(audio))/sr;blend=np.zeros(len(audio))
    for support in intervals:blend=np.maximum(blend,envelope(st,support,policy.ramp_seconds))
    if not np.array_equal((audio*(1-blend)+full*blend).astype(np.float32),baseline):
        raise ValueError('Private native branch does not reproduce original local render')
    zs=hilbert(audio.astype(np.float64));zf=hilbert(full)
    result=baseline.copy();ramps=[]
    for start,end in intervals:
        for side,lo,hi in (('entry',start,start+policy.ramp_seconds),('exit',end-policy.ramp_seconds,end)):
            mask=(st>lo)&(st<hi)&(blend>0)&(blend<1)
            if not mask.any():raise ValueError('Missing sampled ramp')
            joined,gap=polar_ramp(zs[mask],zf[mask],blend[mask])
            result[mask]=joined.real.astype(np.float32)
            ramps.append(dict(side=side,start_seconds=lo,end_seconds=hi,samples=int(mask.sum()),
                max_abs_relative_phase_radians=float(np.max(abs(gap))),
                minimum_branch_analytic_amplitude=float(min(abs(zs[mask]).min(),abs(zf[mask]).min()))))
    if not np.isfinite(result).all() or abs(result).max()>=1:raise ValueError('Nonfinite/clipped phase candidate; no normalization')
    if not np.array_equal(result[blend==0],audio[blend==0]):raise ValueError('Outside samples changed')
    if not np.array_equal(result[blend==1],baseline[blend==1]):raise ValueError('Resynthesized interior changed')
    diagnostic.update(ramps=ramps,baseline_reproduced_exactly=True,interior_samples_exact=True,
        output_time_origin_seconds=diagnostic['source_time_origin_seconds'])
    return result,diagnostic,dict(full=full,blend=blend,baseline_local=baseline)
