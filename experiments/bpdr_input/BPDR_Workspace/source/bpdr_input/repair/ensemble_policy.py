"""Primary support unchanged; final permission uses the full member range."""
import numpy as np
from scipy.ndimage import median_filter
from bpdr_input.repair.held_voice_policy import proposal,spans
from bpdr_input.repair.curve_conditioning import condition_curve
from bpdr_input.repair.repair_controller import plan_fixture


def conditioned_proposal(pair,view,predictions,policy,kind):
    predictions=np.asarray(predictions)
    if (predictions.ndim!=2 or predictions.shape[0]!=3 or not np.isfinite(predictions).all()
            or np.any(predictions[1]>predictions[0]) or np.any(predictions[0]>predictions[2])):
        raise ValueError('Expected ordered median/low/high member envelope')
    if kind!='median11':raise ValueError('Only frozen median11 condition is supported')
    analysis=pair[view];times=analysis['features']['times_seconds']
    # Regionization depends on the primary median only. This preliminary plan
    # cannot authorize rendering: the actual member envelope is checked below.
    detail=proposal(pair['audio_'+view],times,analysis['views'],np.stack([predictions[0]]*3),policy)
    old=detail['plan']
    if not old['permitted'] or old['identity']:return detail['curve'],old,detail['allowed']
    curve=condition_curve(detail['curve'],kind)
    filtered=median_filter(predictions,size=(1,5),mode='nearest')
    low=np.zeros(len(curve),dtype=curve.dtype);high=low.copy()
    for start,stop in spans(detail['curve']!=0):
        # Envelope branches are uncertainty evidence, never authored edits.
        # Apply identical monotone median operators over the PRIMARY support.
        low[start:stop]=median_filter(filtered[1,start:stop],size=11,mode='nearest')
        high[start:stop]=median_filter(filtered[2,start:stop],size=11,mode='nearest')
    changed=curve!=0
    if np.any(low[changed]>curve[changed]) or np.any(high[changed]<curve[changed]):
        raise ValueError('Conditioning broke the member envelope')
    plan=plan_fixture(pair['audio_'+view],22050,times,curve,low,high-low,
                      detail['reliable'],detail['protected'])
    if plan['permitted'] and plan['intervals_seconds']!=old['intervals_seconds']:
        raise ValueError('Ensemble changed primary support')
    return curve,plan,detail['allowed']
