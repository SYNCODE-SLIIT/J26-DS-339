"""Loss and evaluation utilities for notebook 07; no architecture or data changes."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_mask(targets, beat_mask, classes, reserved_ids=(0, 1)):
    """Use both saved masks, exclude reserved labels, retain supervised N."""
    ids = targets['target_chord_ids']
    if ids.shape != beat_mask.shape or ids.shape != targets['target_loss_mask'].shape:
        raise ValueError('Target and beat-mask shapes differ')
    mask = targets['target_loss_mask'] & beat_mask
    if ((ids[mask] < 0) | (ids[mask] >= classes)).any():
        raise ValueError('Active target outside saved vocabulary')
    for index in reserved_ids:
        mask = mask & (ids != index)
    return mask


def count_training_targets(loader, classes, reserved_ids=(0, 1)):
    """Call only with the original training split; no validation/test fitting."""
    counts = torch.zeros(classes, dtype=torch.int64)
    for batch in loader:
        targets = batch['targets']
        mask = supervised_mask(targets, batch['inputs']['beat_attention_mask'], classes, reserved_ids)
        counts += torch.bincount(targets['target_chord_ids'][mask].cpu(), minlength=classes)
    return counts


def sqrt_inverse_weights(counts, alpha=0.5, maximum=5.0):
    """w_c=min(maximum,(N/(K*n_c))**alpha); absent/reserved classes stay zero."""
    if counts.ndim != 1 or (counts < 0).any() or not 0 < alpha <= 1 or maximum <= 0:
        raise ValueError('Invalid counts or weighting settings')
    observed = counts > 0
    if not observed.any():
        raise ValueError('Training contains no supervised targets')
    weights = torch.zeros_like(counts, dtype=torch.float64)
    weights[observed] = (counts.sum().double() / (int(observed.sum()) * counts[observed].double())).pow(alpha).clamp_max(maximum)
    return weights.float()


def weighted_masked_cross_entropy(logits, targets, beat_mask, weights, reserved_ids=(0, 1)):
    if weights.shape != (logits.shape[-1],) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Weights must be a finite nonnegative complete-vocabulary vector')
    mask = supervised_mask(targets, beat_mask, logits.shape[-1], reserved_ids)
    ids = targets['target_chord_ids'][mask]
    weights = weights.to(device=logits.device, dtype=logits.dtype)
    if not ids.numel() or not weights[ids].sum().item():
        return logits.sum() * 0
    return F.cross_entropy(logits[mask], ids, weight=weights)


def metrics_from_confusion(confusion, labels, reserved_ids=(0, 1)):
    """Macro F1 averages supervised classes WITH positive test-reference support.

    Zero-reference classes stay in the detailed report, but not the macro mean.
    F1 is zero when a supported class has no predictions; undefined precision is
    None in tables and treated as zero for computing F1. Reserved predictions
    remain errors; they are never silently replaced by musical candidates.
    """
    if confusion.shape != (len(labels), len(labels)):
        raise ValueError('Confusion dimensions differ from vocabulary')
    support = confusion.sum(1).double()
    predicted = confusion.sum(0).double()
    correct = confusion.diagonal().double()
    precision = correct / predicted.clamp_min(1)
    recall = correct / support.clamp_min(1)
    f1 = 2 * correct / (support + predicted).clamp_min(1)
    eligible = support > 0
    for index in reserved_ids:
        eligible[index] = False
    total = int(support[eligible].sum())
    if not total:
        raise ValueError('No supported supervised classes to evaluate')
    rows = []
    for index, label in enumerate(labels):
        quality = label.split(':', 1)[1].split('/', 1)[0] if ':' in label else label
        rows.append({'target_id':index, 'reference_chord':label, 'quality':quality,
            'reserved':index in reserved_ids, 'support':int(support[index]),
            'predicted_count':int(predicted[index]), 'correct':int(correct[index]),
            'precision':float(precision[index]) if predicted[index] else None,
            'recall':float(recall[index]) if support[index] else None,
            'f1':float(f1[index]) if support[index] or predicted[index] else None,
            'included_in_macro':bool(eligible[index])})
    summary = {'valid_targets':total, 'exact_correct':int(correct[eligible].sum()),
        'accuracy':float(correct[eligible].sum()/total),
        'macro_f1':float(f1[eligible].mean()),
        'weighted_f1':float((f1[eligible]*support[eligible]).sum()/total),
        'macro_classes':int(eligible.sum()),
        'macro_policy':'Positive test-reference support, excluding reserved PAD/UNK; supervised N included.'}
    return summary, rows


def quality_metrics(confusion, labels):
    from experiments.rahul.src.normalize_chords import QUALITY_SUFFIXES
    qualities = [label.split(':',1)[1].split('/',1)[0] if ':' in label else label for label in labels]
    names = set(QUALITY_SUFFIXES) | (set(qualities)-{'PAD','UNK'})
    rows = []
    for quality in sorted(names):
        members = [i for i,name in enumerate(qualities) if name == quality]
        support = sum(int(confusion[i].sum()) for i in members)
        correct = sum(int(confusion[i,i]) for i in members)
        rows.append({'reference_quality':quality, 'support':support, 'exact_correct':correct,
            'exact_chord_accuracy':correct/support if support else None,
            'vocabulary_classes':len(members)})
    return rows
