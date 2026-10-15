"""Four-arm synthetic scoring, called only after source-only predictions seal."""
import numpy as np
from .segment_phase_scoring import errors


def score_four(frequencies,retained,f0,voiced,distance):
    if set(frequencies)!=set('BCDE') or retained.dtype!=bool or voiced.dtype!=bool:raise ValueError('Invalid four-arm domain')
    if any(f.shape!=retained.shape for f in list(frequencies.values())+[f0,voiced,distance]):raise ValueError('Scoring grids differ')
    if not np.all(np.isfinite(f0)&(f0>0)):raise ValueError('Invalid construction truth')
    valid={arm:np.isfinite(f)&(f>0) for arm,f in frequencies.items()}
    domain=retained&voiced;common=domain.copy()
    for mask in valid.values():common&=mask
    result=dict(retained_denominator=int(retained.sum()),retained_voiced_denominator=int(domain.sum()),
        retained_silent_samples=int((retained&~voiced).sum()),common_four_arm_voiced_samples=int(common.sum()),arms={})
    for arm,f in frequencies.items():
        own=domain&valid[arm];common_error=np.full(len(f),np.nan)
        common_error[common]=abs(1200*np.log2(f[common]/f0[common]))
        bins=[]
        for label,lo,hi in [('lt100ms',0,2205),('100to200ms',2205,4410),('200to500ms',4410,11025),('ge500ms',11025,len(f)+1)]:
            mask=(distance>=lo)&(distance<hi)
            bins.append(dict(bin=label,retained_samples=int((retained&mask).sum()),common_voiced_samples=int((common&mask).sum()),
                absolute_error_cents=errors(common_error[common&mask])))
        result['arms'][arm]=dict(invalid_on_retained=int((retained&~valid[arm]).sum()),
            invalid_on_retained_voiced=int((domain&~valid[arm]).sum()),scored_voiced_samples=int(own.sum()),
            own_valid_absolute_error_cents=errors(abs(1200*np.log2(f[own]/f0[own]))),
            common_four_arm_absolute_error_cents=errors(common_error[common]),edge_bins=bins)
    return result


def score_saved(profile,audio,sr,frozen,local,new,islands):
    from .framewise_fixtures import constructed,signal
    _,f0,amp=constructed(profile,sr)
    if not np.array_equal(audio,signal(f0,amp,sr)):raise ValueError('Construction/source mismatch')
    distance=np.full(len(audio),-1,dtype=np.int64)
    for island in islands:
        lo,hi=island['start_sample'],island['end_sample'];i=np.arange(hi-lo)
        distance[lo:hi]=np.minimum(i,hi-lo-1-i)
    return score_four(dict(B=frozen['estimated_frequency_hz'],C=local['local_frequency_hz'],
        D=new['D_frequency_hz'],E=new['E_frequency_hz']),frozen['retained_support'],f0,amp>0,distance)
