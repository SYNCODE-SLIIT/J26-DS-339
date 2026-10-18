"""Read-only geometry/amplitude diagnosis of already sealed phase predictions."""
import numpy as np

VERSION='saved-phase-regression-diagnosis-0.1'


def stats(values):
    values=np.asarray(values)
    if not len(values):return None
    finite=values[np.isfinite(values)]
    return dict(samples=int(len(values)),finite_samples=int(len(finite)),nonfinite_samples=int(len(values)-len(finite)),
                minimum=float(np.min(finite)) if len(finite) else None,median=float(np.median(finite)) if len(finite) else None,
                p95=float(np.quantile(finite,.95)) if len(finite) else None,maximum=float(np.max(finite)) if len(finite) else None)


def runs(mask):
    x=np.flatnonzero(np.diff(np.r_[False,mask,False]))
    return [dict(start_sample=int(lo),end_sample=int(hi),samples=int(hi-lo)) for lo,hi in zip(x[::2],x[1::2])]


def diagnose_saved(frozen,local,islands,global_center_hz,true_f0=None,true_voiced=None):
    """No filtering, mask change, phase alignment, renderer or acceptance decision.

    Construction truth is optional post-seal analysis only. Without it, no
    frequency-accuracy statistic is produced for recordings.
    """
    for k in ('sample_reliable','guarded_support','retained_support'):
        if not np.array_equal(frozen[k],local[k]):raise ValueError('Frozen support changed')
    support=frozen['retained_support'];n=len(support);indices=np.arange(n)
    distance=np.full(n,-1,dtype=np.int64);bad= support & (~np.isfinite(local['local_frequency_hz']) | (local['local_frequency_hz']<=0))
    rows=[]
    for island in islands:
        lo,hi=island['start_sample'],island['end_sample']
        distance[lo:hi]=np.minimum(indices[lo:hi]-lo,hi-1-indices[lo:hi])
        region=(indices>=lo)&(indices<hi);guard=region&frozen['guarded_support'];failure=region&bad
        center=island['local_center_hz']
        row=dict(start_sample=lo,end_sample=hi,samples=hi-lo,status=island['status'],
                 local_center_hz=center,global_center_hz=global_center_hz,
                 center_exactly_equals_global=center==global_center_hz if center is not None else None,
                 center_difference_cents=float(1200*np.log2(center/global_center_hz)) if center is not None else None,
                 retained_samples=int((support&region).sum()),bad_local_on_retained=int(failure.sum()))
        if failure.any():
            b=frozen['estimated_amplitude'];c=local['local_amplitude']
            bm=float(np.median(b[guard]));cm=float(np.median(c[guard]))
            row.update(bad_sample_indices=np.flatnonzero(failure).tolist(),bad_runs=runs(failure),
                local_frequency_on_bad=stats(local['local_frequency_hz'][failure]),
                global_frequency_on_bad=stats(frozen['estimated_frequency_hz'][failure]),
                bad_distance_to_raw_edge_seconds=stats(distance[failure]/22050),
                global_amplitude_on_bad=stats(b[failure]),local_amplitude_on_bad=stats(c[failure]),
                global_guarded_island_amplitude_median=bm,local_guarded_island_amplitude_median=cm,
                global_bad_amplitude_relative_to_guarded_median=stats(b[failure]/bm) if bm>0 else None,
                local_bad_amplitude_relative_to_guarded_median=stats(c[failure]/cm) if cm>0 else None)
        rows.append(row)
    if np.any(support&(distance<0)):raise ValueError('Retained samples outside raw islands')
    result=dict(decoded_samples=n,retained_samples=int(support.sum()),local_bad_on_retained=int(bad.sum()),
                raw_islands=rows,synthetic_error_by_edge_distance=None,
                meaning='Saved estimates and geometry only; low analytic amplitude is not human voicing or established failure causation')
    if true_f0 is not None or true_voiced is not None:
        if true_f0 is None or true_voiced is None or true_f0.shape!=(n,) or true_voiced.shape!=(n,):
            raise ValueError('Both truth grids required for synthetic scoring')
        b=frozen['estimated_frequency_hz'];c=local['local_frequency_hz']
        vb=np.isfinite(b)&(b>0);vc=np.isfinite(c)&(c>0);domain=support&true_voiced;common=domain&vb&vc
        if not np.all(np.isfinite(true_f0)&(true_f0>0)):raise ValueError('Invalid synthetic truth')
        errors={name:np.full(n,np.nan) for name in ('B','C')}
        for name,f in [('B',b),('C',c)]:errors[name][common]=abs(1200*np.log2(f[common]/true_f0[common]))
        bins=[]
        for label,lo,hi in [('lt100ms',0,2205),('100to200ms',2205,4410),('200to500ms',4410,11025),('ge500ms',11025,n+1)]:
            in_bin=(distance>=lo)&(distance<hi);scored=common&in_bin
            bins.append(dict(bin=label,retained_samples=int((support&in_bin).sum()),common_valid_voiced_samples=int(scored.sum()),
                B=stats(errors['B'][scored]),C=stats(errors['C'][scored]),
                local_error_larger_samples=int((scored&(errors['C']>errors['B'])).sum())))
        peaks={}
        for name in ('B','C'):
            if common.any():
                ix=np.flatnonzero(common);peak=int(ix[np.argmax(errors[name][common])])
                peaks[name]=dict(sample=peak,absolute_error_cents=float(errors[name][peak]),
                    distance_to_raw_edge_seconds=float(distance[peak]/22050))
            else:peaks[name]=None
        result['synthetic_error_by_edge_distance']=dict(bins=bins,peak_errors=peaks,
            retained_samples=int(support.sum()),retained_true_voiced_samples=int(domain.sum()),common_valid_voiced_samples=int(common.sum()),
            invalid_B_on_retained=int((support&~vb).sum()),invalid_C_on_retained=int((support&~vc).sum()),
            retained_silence_samples=int((support&~true_voiced).sum()))
    return result
