"""Exact observability at the stored float32 boundary feature grid; no tolerance."""
import numpy as np


def changed_boundary_frames(nominal, altered):
    a=np.asarray(nominal);b=np.asarray(altered)
    if a.ndim!=1 or not len(a) or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid aligned boundary feature arrays')
    # write_views stores boundary_features_71 as float32; metadata alone is not a contrast.
    return int(np.count_nonzero(a.astype(np.float32)!=b.astype(np.float32)))
