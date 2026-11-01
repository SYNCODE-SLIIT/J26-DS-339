"""Training-only candidate labels; none of these inputs enter region generation."""
import numpy as np


def label_candidate(interval,target,known):
    start,stop=interval;target=np.asarray(target);known=np.asarray(known)
    if (target.ndim!=1 or known.shape!=target.shape or known.dtype!=bool
            or not np.isfinite(target).all() or not 0<=start<stop<=len(target)):
        raise ValueError('Invalid candidate supervision')
    mask=known[start:stop];values=target[start:stop][mask]
    if mask.mean()<.5:return None
    if np.all(values==0):return 0.
    positive=values[values!=0]
    if (len(positive)/(stop-start)>=.5 and len(positive)/len(values)>=.95
            and np.all(positive==positive[0]) and abs(positive[0])<=30.):
        return float(positive[0])
    return None
