"""Fixed-parent negative sampling for a declared training fold only.

Preserve class/group/record draws and every positive frame. Replace half the
zero-C and original draws with high-request, unprotected known-zero frames.
This is an experimental sampler, not calibrated confidence or runtime permission.
"""
import numpy as np


def mined_trace(records, trace, predictions, protected, seed):
    trace=np.asarray(trace)
    if trace.ndim!=3 or trace.shape[1:]!=(36,2) or not np.issubdtype(trace.dtype,np.integer):
        raise ValueError('Expected updates x36 x2 integer physical sampler trace')
    if len(predictions)!=len(records) or len(protected)!=len(records):
        raise ValueError('Missing training-record mining evidence')
    pools={}
    for i,record in enumerate(records):
        target=np.asarray(record['target']);mask=np.asarray(record['mask'])
        pred=np.asarray(predictions[i]);guard=np.asarray(protected[i])
        if target.ndim!=1 or mask.shape!=target.shape or pred.shape!=target.shape or guard.shape!=target.shape:
            raise ValueError('Mining evidence frame mismatch')
        if mask.dtype!=bool or guard.dtype!=bool or not np.isfinite(pred).all() or not np.isfinite(target).all():
            raise ValueError('Invalid mining evidence')
        eligible=np.flatnonzero(mask&(target==0)&~guard)
        if not len(eligible):
            pools[i]=np.empty(0,dtype=int);continue
        order=np.lexsort((eligible,-abs(pred[eligible])))
        top=eligible[order[:max(1,int(np.ceil(.20*len(eligible))))]]
        pools[i]=top[abs(pred[top])>=6.]
    result=trace.copy();rng=np.random.default_rng(seed)
    stats=dict(hard_draws=0,changed_frame_draws=0,uniform_fallbacks=0,
               fixed_top_fraction=.20,fixed_request_floor_cents=6.,hard_draws_per_negative_class=6)
    for step,batch in enumerate(trace):
        for slot,(index,frame) in enumerate(batch):
            if not 0<=index<len(records) or not 0<=frame<len(records[index]['target']):
                raise ValueError('Invalid sampler index')
            record=records[index]
            if slot<12:
                if record['kind']!='corrupted' or not record['mask'][frame] or record['target'][frame]==0:
                    raise ValueError('Invalid positive sampler class')
            elif slot<24:
                if record['kind']!='corrupted' or not record['mask'][frame] or record['target'][frame]!=0:
                    raise ValueError('Invalid zero-C sampler class')
            elif record['kind']!='original' or not record['mask'][frame] or record['target'][frame]!=0:
                raise ValueError('Invalid original sampler class')
            if slot not in range(12,18) and slot not in range(24,30):
                continue
            pool=pools[index]
            if not len(pool):
                stats['uniform_fallbacks']+=1;continue
            replacement=int(rng.choice(pool));result[step,slot,1]=replacement
            stats['hard_draws']+=1;stats['changed_frame_draws']+=int(replacement!=frame)
    assert np.array_equal(result[:,:12],trace[:,:12])
    assert np.array_equal(result[:,:,0],trace[:,:,0])
    return result,stats
