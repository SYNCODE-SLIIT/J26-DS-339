"""Experimental curve shaping confined to already permitted sample supports.

No fitting, reference pitch, target interval, scale or permission relaxation.
The original numerical controller must recheck every returned curve.
"""
import numpy as np
from scipy.ndimage import median_filter

from bpdr_input.repair.held_voice_policy import spans, proposal
from bpdr_input.repair.repair_controller import plan_fixture

CONDITIONS = ('unchanged', 'median11', 'span_median')


def condition_curve(curve, kind):
    curve = np.asarray(curve)
    if curve.ndim != 1 or not np.isfinite(curve).all() or kind not in CONDITIONS:
        raise ValueError('Invalid curve or condition')
    result = curve.copy()
    for start, stop in spans(curve != 0):
        one = curve[start:stop]
        if not (np.all(one > 0) or np.all(one < 0)):
            raise ValueError('Curve support crosses correction directions')
        if kind == 'median11':
            result[start:stop] = median_filter(one, size=11, mode='nearest')
        elif kind == 'span_median':
            result[start:stop] = np.median(one)
        # Median statistics cannot increase amplitude beyond observed values,
        # erase the support, bridge a gap or create additional edit duration.
        if np.max(abs(result[start:stop])) > np.max(abs(one)):
            raise ValueError('Conditioning increased maximum amplitude')
    if not np.array_equal(result != 0, curve != 0):
        raise ValueError('Conditioning changed support')
    return result


def conditioned_proposal(pair, view, predictions, policy, kind):
    analysis = pair[view]
    times = analysis['features']['times_seconds']
    detail = proposal(pair['audio_'+view], times, analysis['views'], predictions, policy)
    old_plan = detail['plan']
    if not old_plan['permitted'] or old_plan['identity']:
        return detail['curve'], old_plan, detail['allowed']
    curve = condition_curve(detail['curve'], kind)
    # Apply the identical transformation to all metadata views over the same
    # already-approved support. These are boundary perturbations, not calibrated
    # confidence estimates; the model currently excludes boundary metadata.
    changed = detail['curve'] != 0
    alternate = condition_curve(np.where(changed, detail['predictions'][1], 0.), kind)
    other = condition_curve(np.where(changed, detail['predictions'][2], 0.), kind)
    disagreement = np.maximum(abs(curve-alternate), abs(curve-other))
    new_plan = plan_fixture(pair['audio_'+view], 22050, times, curve, alternate,
                            disagreement, detail['reliable'], detail['protected'])
    if new_plan['permitted'] and new_plan['intervals_seconds'] != old_plan['intervals_seconds']:
        raise ValueError('Conditioning changed permission intervals')
    return curve, new_plan, detail['allowed']
