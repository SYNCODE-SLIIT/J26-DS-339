"""Versioned input-only held-voice eligibility and multi-candidate selection.

Pitch-only provisional boundaries do not veto the small local deviations this
scope is meant to assess. Unreliable voice, energy onsets, large transitions,
user-protected samples and the existing numerical/rendering limits still veto.
"""
from dataclasses import asdict,dataclass
import numpy as np
from scipy.ndimage import median_filter
from bpdr_input.repair.repair_controller import plan_fixture


@dataclass(frozen=True)
class HeldPolicy:
    version:str='held-voice-input-policy-1'
    activation_cents:float=12.
    release_fraction:float=.5
    gap_frames:int=0
    region_split_cents:float|None=None
    minimum_step_fit_gain:float|None=None


def spans(mask):
    delta=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(delta==1),np.flatnonzero(delta==-1)))


def eligibility(raw,user_protected=None,region_split_cents=None):
    raw=np.asarray(raw)
    if raw.ndim!=2 or raw.shape[1]!=71 or not np.isfinite(raw).all():
        raise ValueError('Invalid runtime input')
    reliable=raw[:,66]>.5
    protected=np.zeros(len(raw),bool)
    # Fundamental-frequency discontinuities >=70c are outside the declared
    # +/-30c repair scope. Smaller pitch-only steps are not acoustic onsets.
    changes=np.r_[True,np.diff(reliable.astype(int))!=0]
    jump=np.r_[False,(np.abs(np.diff(raw[:,64]))>=70.)&reliable[1:]&reliable[:-1]]
    energy=np.r_[False,(np.abs(np.diff(raw[:,68]))>=6.)&reliable[1:]&reliable[:-1]]
    for i in np.flatnonzero(changes|jump|energy):
        protected[max(0,i-5):min(len(raw),i+6)]=True
    if user_protected is not None:
        user_protected=np.asarray(user_protected)
        if user_protected.shape!=protected.shape or user_protected.dtype!=bool:
            raise ValueError('Invalid user protection mask')
        protected|=user_protected
    allowed=np.zeros(len(raw),bool)
    regions=spans(reliable)
    if region_split_cents is not None:
        if region_split_cents!=70.:raise ValueError('Unregistered acoustic-region threshold')
        from bpdr_input.repair.rule_correction import segment_notes,RulePolicy
        regions=segment_notes(raw[:,64],reliable,RulePolicy(note_split_cents=region_split_cents))
        for start,stop in regions:
            if start>0:protected[max(0,start-5):min(len(raw),start+6)]=True
            if stop<len(raw):protected[max(0,stop-5):min(len(raw),stop+6)]=True
    for start,stop in regions:
        if (stop-start)*.02<1.10:continue
        y=raw[start:stop,64];x=np.arange(len(y))
        spread=np.std(y-np.polyval(np.polyfit(x,y,1),x))
        if spread<=20.: allowed[start:stop]=True
    return allowed,reliable,protected


def proposal(audio,times,raw_views,predictions,policy=HeldPolicy(),user_protected=None):
    predictions=np.asarray(predictions)
    if predictions.shape!=(3,len(times)) or not np.isfinite(predictions).all():
        raise ValueError('Invalid aligned boundary-view predictions')
    if policy.activation_cents<=0 or not 0<policy.release_fraction<=1 or policy.gap_frames not in (0,2,4):
        raise ValueError('Invalid policy')
    allowed,reliable,protected=eligibility(raw_views[0],user_protected,policy.region_split_cents)
    # Fixed 100ms median reduces single-frame jitter; no amplitude clipping.
    filtered=median_filter(predictions,size=(1,5),mode='nearest')
    primary=filtered[0];curve=np.zeros(len(times));candidates=[]
    step_measurements=None
    if policy.minimum_step_fit_gain is not None:
        if not .65<=policy.minimum_step_fit_gain<=1.:raise ValueError('Invalid step-fit guard')
        from bpdr_input.modeling.step_context import evidence
        step_measurements=evidence(raw_views[0])
    release=policy.activation_cents*policy.release_fraction
    for sign in (-1,1):
        support=(primary*sign>=release)&allowed&reliable
        if policy.gap_frames:
            for start,stop in spans(~support):
                if (start>0 and stop<len(support) and stop-start<=policy.gap_frames
                    and support[start-1] and support[stop]
                    and np.all(primary[start:stop]*sign>=release*.5)
                    and np.all(allowed[start:stop]&reliable[start:stop]&~protected[start:stop])):
                    support[start:stop]=True
        for start,stop in spans(support):
            reason=None
            step_gain=float(step_measurements[start:stop,1].max()) if step_measurements is not None else None
            if not np.any(primary[start:stop]*sign>=policy.activation_cents):reason='NO_ACTIVATION'
            elif np.any(protected[start:stop]):reason='PROTECTED_REGION'
            elif (stop-start)*.02<.86-1e-12:reason='INSUFFICIENT_SUPPORT'
            elif step_gain is not None and step_gain<policy.minimum_step_fit_gain:reason='STEP_SHAPE_EVIDENCE'
            candidates.append(dict(start=int(start),stop=int(stop),reason=reason,step_fit_gain=step_gain))
            if reason is None:curve[start:stop]=primary[start:stop]
    disagreement=np.max(np.abs(filtered[1:]-filtered[0]),axis=0)
    plan=plan_fixture(audio,22050,times,curve,filtered[1],disagreement,reliable,protected)
    return dict(curve=curve,plan=plan,allowed=allowed,reliable=reliable,protected=protected,
        predictions=filtered,policy=asdict(policy),candidates=candidates)
