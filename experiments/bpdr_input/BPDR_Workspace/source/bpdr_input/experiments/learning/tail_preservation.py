"""Fixed upper-third known-zero loss, experimental and training-only.

Adapt average top-k aggregation per existing negative class. Inputs are normalized
corrections (cents/30); class coefficients and the twelve rows per class stay fixed.
No uncertainty, runtime permission or natural intended-pitch guarantee follows.
"""
import torch


def upper_tail_preservation(zero_c,original):
    for values in (zero_c,original):
        if values.shape!=(12,) or not torch.is_floating_point(values) or not torch.isfinite(values).all():
            raise ValueError('Expected twelve finite normalized corrections per negative class')
    return .5*torch.topk(zero_c.square(),4,sorted=False).values.mean()+.5*torch.topk(original.square(),4,sorted=False).values.mean()
