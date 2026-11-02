"""Diagnostic float64 version of PESTO's unchanged argmax-local weighted mean."""
import numpy as np


def decode_float64(activations):
    a=np.asarray(activations)
    if a.ndim!=2 or a.shape[1]!=384 or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('Expected finite nonnegative384-bin frame distributions')
    indices=np.clip(np.argmax(a,axis=-1)[:,None]+np.arange(-2,3),0,383)
    local=np.take_along_axis(a,indices,axis=-1).astype(np.float64)
    total=local.sum(axis=-1)
    if np.any(total<=0):raise ValueError('Empty decoder window')
    pitch=(local*(indices.astype(np.float64)/3.)).sum(axis=-1)/total
    return 440.*np.exp2((pitch-69.)/12.)
