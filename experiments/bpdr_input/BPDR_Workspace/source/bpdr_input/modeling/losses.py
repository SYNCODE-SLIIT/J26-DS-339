"""Phase 3 objectives: supervised anchors, smoothness and the two BPDR response terms."""
from dataclasses import dataclass
import torch

CONDITIONS = ('augmentation_only', 'frame_only', 'bpdr')


@dataclass(frozen=True)
class LossWeights:
    # Provisional starting values; the proposal selects them on development data.
    boundary: float = 1.
    detuning: float = 1.
    smoothness: float = .1


def masked_mean_abs(a, b, mask):
    """d_M: mean absolute difference over mask, averaged over examples whose mask is not empty.

    Returns the value and the number of examples left out because their mask was empty.
    """
    mask = mask.to(a.dtype)
    frames = mask.sum(dim=1)
    used = frames > 0
    if not used.any():
        return (a*0).sum(), int(len(frames))
    per_example = ((a-b).abs()*mask).sum(dim=1)[used]/frames[used]
    return per_example.mean(), int((~used).sum())


def smoothness(correction, voiced):
    """L_S: abrupt changes of the correction itself between neighbouring voiced frames, never the sung contour."""
    return masked_mean_abs(correction[:, 1:], correction[:, :-1], voiced[:, 1:] & voiced[:, :-1])


def boundary_invariance(r, mask):
    """L_B: the same audio under changed boundary metadata should get the same repair."""
    ab, skipped_ab = masked_mean_abs(r['A'], r['B'], mask)
    cd, skipped_cd = masked_mean_abs(r['C'], r['D'], mask)
    return .5*(ab+cd), skipped_ab+skipped_cd


def counter_detuning(r, detuning, mask):
    """L_D: adding local detuning e should change the repair by -e, and by nothing outside its support."""
    ca, skipped_ca = masked_mean_abs(r['C']-r['A'], -detuning, mask)
    db, skipped_db = masked_mean_abs(r['D']-r['B'], -detuning, mask)
    return .5*(ca+db), skipped_ca+skipped_db


def objective(r, batch, condition, weights=LossWeights(), response=True):
    """Total loss for one condition. `response=False` is the supervised warm-up shared by all three."""
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown condition: {condition}')
    mask, detuning = batch['mask'], batch['detuning']
    zero = torch.zeros_like(detuning)
    parts, skipped = {}, {}
    # L_A: zero for accepted originals, the compensating target for verified detuned views.
    anchors = [masked_mean_abs(r[view], zero if view in 'AB' else -detuning, mask) for view in 'ABCD']
    parts['anchor'], skipped['anchor'] = sum(a[0] for a in anchors)/4, sum(a[1] for a in anchors)
    smooth = [smoothness(r[view], batch['voiced']['A' if view in 'AB' else 'C']) for view in 'ABCD']
    parts['smoothness'], skipped['smoothness'] = sum(s[0] for s in smooth)/4, sum(s[1] for s in smooth)
    total = parts['anchor']+weights.smoothness*parts['smoothness']
    if condition == 'bpdr' and response:
        parts['boundary'], skipped['boundary'] = boundary_invariance(r, mask)
        parts['detuning'], skipped['detuning'] = counter_detuning(r, detuning, mask)
        total = total+weights.boundary*parts['boundary']+weights.detuning*parts['detuning']
    return total, parts, skipped
