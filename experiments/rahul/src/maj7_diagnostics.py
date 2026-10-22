"""Read-only major-seventh prediction diagnostics; no training operations."""
from __future__ import annotations

import torch


def chord_parts(label: str) -> tuple[str | None, str | None]:
    if ':' not in label:
        return None, None
    root, detail = label.split(':', 1)
    return root, detail.split('/', 1)[0]


def binary_quality_metrics(true_positive: int, false_positive: int, false_negative: int) -> dict:
    tp, fp, fn = true_positive, false_positive, false_negative
    return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn,
        'precision':tp/(tp+fp) if tp+fp else None,
        'recall':tp/(tp+fn) if tp+fn else None,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'definition':'Binary maj7-quality detection across all valid beats, ignoring root and inversion.'}


def diagnose_logits(logits: torch.Tensor, target_id: int, labels: list[str]) -> dict:
    """Raw saved-class softmax; no PAD/UNK filtering or probability renormalization.

    Exact ties are broken by saved vocabulary order for deterministic ranks.
    """
    if logits.ndim != 1 or len(logits) != len(labels) or not 0 <= target_id < len(labels):
        raise ValueError('Logits, target and vocabulary disagree')
    if not torch.isfinite(logits).all():
        raise ValueError('Non-finite logits')
    probabilities = logits.softmax(-1)
    order = logits.argsort(descending=True, stable=True)
    rank = int(torch.nonzero(order == target_id)[0, 0])+1
    prediction = int(order[0])
    root, _ = chord_parts(labels[target_id])
    predicted_root, predicted_quality = chord_parts(labels[prediction])
    maj7_ids = [i for i,label in enumerate(labels) if chord_parts(label)[1]=='maj7']
    same_root_ids = [i for i in maj7_ids if chord_parts(labels[i])[0]==root]
    triad = f'{root}:maj'
    triad_id = labels.index(triad) if triad in labels else None
    return {'predicted_id':prediction, 'predicted_label':labels[prediction],
        'exact_rank':rank, 'top1_correct':rank<=1, 'top3_correct':rank<=3, 'top5_correct':rank<=5,
        'reference_probability':float(probabilities[target_id]),
        'winner_probability':float(probabilities[prediction]),
        'winner_minus_reference_logit':float(logits[prediction]-logits[target_id]),
        'all_maj7_probability_mass':float(probabilities[maj7_ids].sum()),
        'same_root_maj7_probability_mass':float(probabilities[same_root_ids].sum()),
        'plain_major_triad':triad,
        'plain_major_triad_probability':float(probabilities[triad_id]) if triad_id is not None else None,
        'predicted_corresponding_plain_triad':labels[prediction]==triad,
        'predicted_same_root_major_any_inversion':predicted_root==root and predicted_quality=='maj',
        'top5':[{'label':labels[int(i)],'probability':float(probabilities[i])} for i in order[:5]]}


def melody_features(inputs: dict, batch: int, position: int, root_pc: int) -> dict:
    """Inspect actual active note slots and root-relative pitch-class cues.

    A major-seventh pitch class is a potential melody cue, not proof of harmony.
    """
    mask = inputs['melody_note_mask'][batch,position]
    tokens = inputs['melody_pitch_ids'][batch,position][mask].tolist()
    midi = [int(value)-2 if value>=2 else None for value in tokens]
    intervals = [(pitch-root_pc)%12 if pitch is not None else None for pitch in midi]
    sequence_mask = inputs['melody_note_mask'][batch] & inputs['beat_attention_mask'][batch].unsqueeze(-1)
    sequence_tokens = inputs['melody_pitch_ids'][batch][sequence_mask]
    valid_tokens = sequence_tokens[sequence_tokens>=2]
    sequence_intervals = (valid_tokens-2-root_pc)%12
    return {'note_count':len(tokens), 'pitch_tokens':tokens, 'rounded_midi':midi,
        'relative_pitch_classes_to_reference_root':intervals,
        'melody_timing':inputs['melody_timing'][batch,position][mask].tolist(),
        'silent':bool(inputs['melody_silent'][batch,position]),
        'major_seventh_pc_in_beat':11 in intervals,
        'major_seventh_pc_in_sequence':bool((sequence_intervals==11).any()),
        'triad_only_pitched_beat':bool(intervals) and all(x in (0,4,7) for x in intervals),
        'cue_note':'Relative pitch class 11 is a possible major-seventh cue; melody non-chord tones and pseudo-label errors remain possible.'}
