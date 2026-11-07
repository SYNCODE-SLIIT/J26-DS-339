"""Three-estimator short-support measurement hypothesis; never production approval."""
from dataclasses import replace,asdict
import numpy as np
import parselmouth
from bpdr_input.repair.interventions import AuditPolicy
from bpdr_input.repair.framewise_audit import audit_curve

POLICY=replace(AuditPolicy(),version='three-tracker-short-audit-hypothesis-1',
               ramp_seconds=.02,analysis_guard_seconds=.07,min_audit_frames=3)


def cross_correlation_pitch(audio,sr,times):
    """Separate waveform remeasurement, not a renderer pitch-tier readout.

    Praat CC shares a library with the renderer and uses temporal path finding;
    it is corroboration, not wholly independent ground truth. No interpolation
    across unvoiced frames. Native10ms samples are matched to the20ms audit grid.
    """
    if sr!=22050:raise ValueError('Fixed sample rate required')
    pitch=parselmouth.Sound(np.asarray(audio,dtype=np.float64),sampling_frequency=sr).to_pitch_cc(
        time_step=.01,pitch_floor=50.,pitch_ceiling=1500.,very_accurate=True)
    native_times=pitch.xs();f0=pitch.selected_array['frequency']
    indices=np.clip(np.searchsorted(native_times,times),0,len(native_times)-1)
    prev=np.maximum(indices-1,0)
    indices=np.where(abs(native_times[prev]-times)<abs(native_times[indices]-times),prev,indices)
    close=abs(native_times[indices]-times)<=.005+1e-12
    return np.where(close&(f0[indices]>0),f0[indices],np.nan)


def audit_short_curve(before,after,yin_before,yin_after,cc_before,cc_after,reliable,intervals,requested):
    times=before['times_seconds'];requested=np.asarray(requested)
    cc_before=np.asarray(cc_before);cc_after=np.asarray(cc_after)
    if cc_before.shape!=times.shape or cc_after.shape!=times.shape:raise ValueError('CC audit grid mismatch')
    result=audit_curve(before,after,yin_before,yin_after,reliable,intervals,requested,POLICY)
    plateau=np.zeros(len(times),bool)
    for start,end in intervals:
        plateau|=(times>=start+POLICY.ramp_seconds+POLICY.analysis_guard_seconds)&(times<=end-POLICY.ramp_seconds-POLICY.analysis_guard_seconds)
    plateau&=reliable
    valid=plateau&before['voiced_mask']&after['voiced_mask']
    valid&=np.isfinite(yin_before)&np.isfinite(yin_after)&(np.asarray(yin_before)>0)&(np.asarray(yin_after)>0)
    valid&=np.isfinite(cc_before)&np.isfinite(cc_after)&(cc_before>0)&(cc_after>0)
    delta=np.full(len(times),np.nan);delta[valid]=1200*np.log2(cc_after[valid]/cc_before[valid])
    coverage=float(valid.sum()/plateau.sum()) if plateau.any() else 0.
    error=abs(delta[valid]-requested[valid])
    median=float(np.median(error)) if len(error) else None
    p90=float(np.quantile(error,.9)) if len(error) else None
    direction=valid&(abs(requested)>=10.)
    wrong=int(np.sum(delta[direction]*requested[direction]<=0))
    checks=[('CC_COMMON_FRAMES',valid.sum()>=3),('CC_COVERAGE',coverage>=.75),
        ('CC_MEDIAN_ERROR',median is not None and median<=5.),
        ('CC_P90_ERROR',p90 is not None and p90<=10.),('CC_DIRECTION',wrong==0)]
    reasons=result['reasons']+[name for name,passed in checks if not passed]
    result.update(version=POLICY.version,passed=not reasons,reasons=reasons,
        cc=dict(frames=int(valid.sum()),coverage=coverage,median_error_cents=median,
                p90_error_cents=p90,wrong_direction_frames=wrong),
        three_estimator_common_frames=int(valid.sum()),experimental=True,
        production_eligible=False,
        limitation='Minimum3 native20ms common observations is a prospectively tested hypothesis, not independent samples or a confidence guarantee. Requires empirical positive/negative-control calibration. Praat CC corroborates YIN/PESTO but shares renderer library; no intended-pitch claim.')
    return result
