"""Candidate audit for changing curves; preserves v0.3 error/coverage limits."""
from dataclasses import asdict
import numpy as np
from .interventions import AuditPolicy, cents_difference


def audit_curve(before,after,yin_before,yin_after,reliable,intervals,requested,policy=AuditPolicy()):
    times=before['times_seconds'];requested=np.asarray(requested)
    if (not np.array_equal(times,after['times_seconds']) or requested.shape!=times.shape
            or not np.isfinite(requested).all() or len(yin_before)!=len(times) or len(yin_after)!=len(times)):
        raise ValueError('Framewise audit grid/curve mismatch')
    plateau=np.zeros(len(times),dtype=bool);outside=reliable.copy()
    for start,end in intervals:
        plateau|=(times>=start+policy.ramp_seconds+policy.analysis_guard_seconds)&(times<=end-policy.ramp_seconds-policy.analysis_guard_seconds)
        outside&=(times<start-policy.analysis_guard_seconds)|(times>end+policy.analysis_guard_seconds)
    plateau&=reliable
    pesto,pvalid=cents_difference(after['f0_hz'],before['f0_hz'],before['voiced_mask']&after['voiced_mask'])
    yin,yvalid=cents_difference(yin_after,yin_before,reliable)
    common=plateau&pvalid&yvalid;outside_valid=outside&pvalid&yvalid
    direction=common&(abs(requested)>=10.)
    def metrics(delta):
        errors=abs(delta[common]-requested[common])
        return dict(frames=int(common.sum()),median_error_cents=float(np.median(errors)),p90_error_cents=float(np.quantile(errors,.9))) if common.any() else None
    fine,coarse=metrics(yin),metrics(pesto)
    pc=float(common.sum()/plateau.sum()) if plateau.any() else 0.
    oc=float(outside_valid.sum()/outside.sum()) if outside.any() else 0.
    median=max(float(np.median(abs(delta[outside_valid]))) for delta in (pesto,yin)) if outside_valid.any() else None
    p90=max(float(np.quantile(abs(delta[outside_valid]),.9)) for delta in (pesto,yin)) if outside_valid.any() else None
    fine_wrong=int(np.count_nonzero(yin[direction]*requested[direction]<=0))
    coarse_wrong=int(np.count_nonzero(pesto[direction]*requested[direction]<0))
    reasons=[]
    for name,ok in [('PLATEAU_FRAME_COUNT',common.sum()>=policy.min_audit_frames),
                    ('PLATEAU_COVERAGE',pc>=policy.min_plateau_coverage_fraction),
                    ('OUTSIDE_FRAME_COUNT',outside_valid.sum()>=policy.min_outside_audit_frames),
                    ('OUTSIDE_COVERAGE',oc>=policy.min_outside_coverage_fraction),
                    ('FINE_MEDIAN_ERROR',fine is not None and fine['median_error_cents']<=policy.median_error_tolerance_cents),
                    ('FINE_P90_ERROR',fine is not None and fine['p90_error_cents']<=policy.p90_error_tolerance_cents),
                    ('COARSE_MEDIAN_ERROR',coarse is not None and coarse['median_error_cents']<=policy.pesto_median_coarse_tolerance_cents),
                    ('COARSE_P90_ERROR',coarse is not None and coarse['p90_error_cents']<=policy.pesto_p90_coarse_tolerance_cents),
                    ('FINE_DIRECTION',fine_wrong==0),('COARSE_DIRECTION',coarse_wrong==0),
                    ('OUTSIDE_MEDIAN_DRIFT',median is not None and median<=policy.outside_pitch_drift_tolerance_cents),
                    ('OUTSIDE_P90_DRIFT',p90 is not None and p90<=policy.outside_p90_pitch_drift_tolerance_cents)]:
        if not ok: reasons.append(name)
    return dict(version='framewise-render-audit-0.1',base_limits=asdict(policy),passed=not reasons,reasons=reasons,
        eligible_plateau_frames=int(plateau.sum()),common_frames=int(common.sum()),plateau_coverage_fraction=pc,
        outside_frames=int(outside_valid.sum()),outside_coverage_fraction=oc,fine=fine,coarse=coarse,
        outside_median_drift_cents=median,outside_p90_drift_cents=p90,
        fine_wrong_direction_frames=fine_wrong,coarse_wrong_direction_frames=coarse_wrong,
        direction_checked_frames=int(direction.sum()),
        direction_rule='Every common frame with abs(requested)>=10 cents must have correct fine direction and nonopposing coarse direction; stricter than constant median-direction check',
        meaning='Changing requested curves compared per frame; ramps/guards excluded. Not musical approval.')
