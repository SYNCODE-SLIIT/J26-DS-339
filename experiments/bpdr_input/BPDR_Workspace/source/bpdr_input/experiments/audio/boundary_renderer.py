"""Experimental boundary-taper candidate, separate from active renderer/policy."""
import numpy as np
from bpdr_input.repair.framewise_renderer import RendererPolicy, render_candidate
from bpdr_input.repair.interventions import envelope
from bpdr_input.repair.repair_controller import check_source

VERSION = 'praat-boundary-taper-candidate-0.1'


def render_boundary_candidate(audio, sr, times, correction_cents, intervals, policy=RendererPolicy()):
    """Taper the 20-ms supplied cents curve, then use the unchanged local renderer.

    Validate the original request before multiplication: tapering must never hide
    an over-limit value, an invalid grid or an outside-support correction. This is
    a boundary-handling hypothesis, not phase alignment or a production policy.
    """
    check_source(audio,sr)
    times=np.asarray(times);curve=np.asarray(correction_cents)
    expected=np.arange(len(audio)//441+1)*.02
    if (times.shape!=expected.shape or not np.allclose(times,expected,rtol=0,atol=1e-12)
            or curve.shape!=times.shape or not np.isfinite(curve).all()
            or np.max(abs(curve))>policy.max_cents):
        raise ValueError('Invalid original correction grid or over-limit curve; no clipping')
    intervals=tuple((float(start),float(end)) for start,end in intervals)
    covered=np.zeros(len(times),dtype=bool);taper=np.zeros(len(times));previous=0.
    for start,end in intervals:
        if (not np.isfinite(start+end) or start<previous or not 0<=start<end<=len(audio)/sr
                or end-start<policy.min_support_seconds-1e-12):
            raise ValueError('Invalid original supports')
        previous=end;covered|=(times>=start)&(times<end)
        taper=np.maximum(taper,envelope(times,(start,end),policy.ramp_seconds))
    if np.any((curve!=0)&~covered):raise ValueError('Original correction outside permitted support')
    tapered=curve*taper
    output,diagnostic=render_candidate(audio,sr,times,tapered,intervals,policy)
    diagnostic.update(candidate_version=VERSION,
        boundary_method='Supplied 20-ms cents curve multiplied by the existing 80-ms raised-cosine envelope before native-tier interpolation',
        boundary_correction_is_not_linear_waveform_pitch=True,
        original_request_validated=True,phase_alignment_implemented=False)
    return output,diagnostic,tapered
