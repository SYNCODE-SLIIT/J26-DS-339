"""Training-only full-pass original query scheduling, no new labels or risk score."""
import numpy as np


def covered_original_trace(records,trace,seed):
    trace=np.asarray(trace)
    if trace.ndim!=3 or trace.shape[1:]!=(36,2) or not np.issubdtype(trace.dtype,np.integer):
        raise ValueError('Expected updates x36 x2 integer trace')
    eligible=[]
    for index,record in enumerate(records):
        target=np.asarray(record['target']);mask=np.asarray(record['mask'])
        if target.ndim!=1 or mask.shape!=target.shape or mask.dtype!=bool or not np.isfinite(target).all():raise ValueError('Invalid training targets/masks')
        if record['kind']=='original':
            if np.any(target[mask]!=0):raise ValueError('Original target must be zero')
            eligible.extend((index,int(frame)) for frame in np.flatnonzero(mask))
    if not eligible:raise ValueError('No eligible original frames')
    budget=len(trace)*12
    if len(eligible)>budget:raise ValueError('Complete original coverage exceeds fixed query budget')
    for batch in trace:
        for slot,(index,frame) in enumerate(batch):
            if not 0<=index<len(records) or not 0<=frame<len(records[index]['target']):raise ValueError('Invalid sampler index')
            record=records[index]
            if not record['mask'][frame]:raise ValueError('Unlabelled sampled frame')
            if slot<12:
                if record['kind']!='corrupted' or record['target'][frame]==0:raise ValueError('Invalid positive class')
            elif slot<24:
                if record['kind']!='corrupted' or record['target'][frame]!=0:raise ValueError('Invalid C-zero class')
            elif record['kind']!='original' or record['target'][frame]!=0:raise ValueError('Invalid original class')
    pool=np.asarray(eligible,dtype=trace.dtype);rng=np.random.default_rng(seed);draws=[];remaining=budget
    while remaining:
        take=pool[rng.permutation(len(pool))][:remaining];draws.append(take);remaining-=len(take)
    schedule=np.concatenate(draws);result=trace.copy();result[:,24:]=schedule.reshape(len(trace),12,2)
    assert np.array_equal(result[:,:24],trace[:,:24])
    assert len(np.unique(schedule,axis=0))==len(eligible)
    return result,dict(original_query_budget=budget,eligible_original_frames=len(eligible),unique_original_frames_queried=len(eligible),
        repeated_draws_after_first_full_pass=budget-len(eligible),permutation_seed=seed,complete_query_coverage=True)
