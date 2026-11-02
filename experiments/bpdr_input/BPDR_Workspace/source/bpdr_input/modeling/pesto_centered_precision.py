"""Numerically centered form of the same five-bin PESTO ALWA estimator."""
import numpy as np


def decode_centered(activations):
    a=np.asarray(activations)
    if a.ndim!=2 or a.shape[1]!=384 or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('Expected finite nonnegative384-bin frame distributions')
    center=np.argmax(a,axis=-1)
    indices=np.clip(center[:,None]+np.arange(-2,3),0,383)
    local=np.take_along_axis(a,indices,axis=-1).astype(np.float64)
    total=local.sum(axis=-1)
    if np.any(total<=0):raise ValueError('Empty decoder window')
    # Weighted mean = center + weighted signed displacement. Avoid multiplying
    # the dominant weight by a large absolute pitch then dividing it away.
    relative=(local*(indices-center[:,None])).sum(axis=-1)/total
    # A440 is bin207. Subtract that exact integer BEFORE adding the small
    # residual; do not form a large MIDI value and subtract69 afterward.
    exponent=((center.astype(np.float64)-207.)+relative)/36.
    return 440.*np.exp2(exponent)
