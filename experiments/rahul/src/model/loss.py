"""Loss only over supervised real beats; ambiguous beats remain attention context."""
import torch
from torch.nn import functional as F


def masked_cross_entropy(logits, targets, beat_attention_mask):
    ids = targets['target_chord_ids']
    valid = targets['target_loss_mask'] & beat_attention_mask
    if logits.shape[:2] != ids.shape or valid.shape != ids.shape:
        raise ValueError('Logits, targets and masks must have matching beat dimensions')
    selected = ids[valid]
    if selected.numel() == 0:
        return logits.sum() * 0  # Differentiable zero; no NaN on empty batches.
    if ((selected < 0) | (selected >= logits.shape[-1])).any():
        raise ValueError('Active target ID outside saved vocabulary range')
    return F.cross_entropy(logits[valid], selected)
