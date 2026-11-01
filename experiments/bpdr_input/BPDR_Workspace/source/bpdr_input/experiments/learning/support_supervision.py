"""Training-only supervision of genuinely labelled contiguous repair support."""
import numpy as np
import torch
from torch.nn import functional as F

BLOCK_FRAMES = 43


def block_starts(target, known):
    """No padding, bridging, ramps, missing labels or sign changes in blocks."""
    target, known = np.asarray(target), np.asarray(known)
    if (target.ndim != 1 or known.shape != target.shape or known.dtype != bool
            or not np.isfinite(target).all()):
        raise ValueError('Invalid training labels')
    starts = []
    for i in range(len(target)-BLOCK_FRAMES+1):
        y = target[i:i+BLOCK_FRAMES]
        if known[i:i+BLOCK_FRAMES].all() and y[0] != 0 and np.all(y == y[0]):
            starts.append(i)
    return np.asarray(starts, dtype=int)


def block_objective(output, logits, target):
    """Dense paired labels plus a conservative soft-min support margin.

    Labels are used only in fitting. No interval is supplied at inference.
    The margin uses the already fixed activation8/release4-cent levels.
    Soft min/max bound the true extrema; an isolated gap cannot hide in a mean.
    """
    if (output.shape != (2, BLOCK_FRAMES) or logits.shape != output.shape
            or target.shape != output.shape or not torch.isfinite(target).all()
            or torch.any(target[0] != 0) or target[1, 0] == 0
            or torch.any(target[1] != target[1, 0]) or torch.any(abs(target) > 30)):
        raise ValueError('Block must contain fully known paired clean/constant-shift labels')
    weights = target.new_tensor([4., 1.])[:, None]
    anchor = (((output-target)/30.).square()*weights).mean()
    response = ((output[1]-output[0]-target[1])/30.).square().mean()
    activity = F.binary_cross_entropy_with_logits(logits, (target != 0).float())
    temperature = .5
    signed = output[1]*torch.sign(target[1, 0])
    minimum = -temperature*torch.logsumexp(-signed/temperature, dim=0)
    maximum_clean = temperature*torch.logsumexp(abs(output[0])/temperature, dim=0)
    margin = F.relu((8.-minimum)/30.).square()+F.relu((maximum_clean-4.)/30.).square()
    return anchor+.25*response+.5*activity+.1*margin, margin
