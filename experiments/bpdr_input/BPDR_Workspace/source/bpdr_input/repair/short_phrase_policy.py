"""Experimental input-only short-support proposals, without production permission."""
from dataclasses import dataclass,asdict,replace
import numpy as np
from scipy.ndimage import median_filter
from bpdr_input.repair.held_voice_policy import spans
from bpdr_input.repair.repair_controller import FixturePolicy,plan_fixture


@dataclass(frozen=True)
class ShortPolicy:
    version:str='short-phrase-input-policy-1'
    activation_cents:float=12.
    release_cents:float=6.
    min_support_seconds:float=.24
    protected_guard_frames:int=4
    median_frames:int=3


def proposal(audio,times,raw_views,predictions,policy=ShortPolicy(),user_protected=None):
    raw=np.asarray(raw_views[0]);predictions=np.asarray(predictions)
    if raw.ndim!=2 or raw.shape[1]!=71 or not np.isfinite(raw).all():raise ValueError('Invalid acoustic input')
    if predictions.shape!=(3,len(times)) or len(raw)!=len(times) or not np.isfinite(predictions).all():raise ValueError('Invalid prediction grid')
    if policy!=ShortPolicy():raise ValueError('Unregistered diagnostic short policy')
    reliable=raw[:,66]>.5
    changes=np.r_[True,np.diff(reliable.astype(int))!=0]
    jump=np.r_[False,(abs(np.diff(raw[:,64]))>=70.)&reliable[1:]&reliable[:-1]]
    energy=np.r_[False,(abs(np.diff(raw[:,68]))>=6.)&reliable[1:]&reliable[:-1]]
    protected=np.zeros(len(raw),bool)
    for i in np.flatnonzero(changes|jump|energy):
        protected[max(0,i-policy.protected_guard_frames):min(len(raw),i+policy.protected_guard_frames+1)]=True
    if user_protected is not None:
        user_protected=np.asarray(user_protected)
        if user_protected.shape!=protected.shape or user_protected.dtype!=bool:raise ValueError('Invalid user protection')
        protected|=user_protected
    filtered=median_filter(predictions,size=(1,policy.median_frames),mode='nearest')
    curve=np.zeros(len(times));candidates=[]
    for sign in [-1,1]:
        for a,b in spans((filtered[0]*sign>=policy.release_cents)&reliable):
            reason=None
            if not np.any(filtered[0,a:b]*sign>=policy.activation_cents):reason='NO_ACTIVATION'
            elif np.any(protected[a:b]):reason='PROTECTED_REGION'
            elif (b-a)*.02<policy.min_support_seconds-1e-12:reason='INSUFFICIENT_SUPPORT'
            candidates.append(dict(start=int(a),stop=int(b),reason=reason))
            if reason is None:curve[a:b]=filtered[0,a:b]
    disagreement=np.max(abs(filtered[1:]-filtered[0]),axis=0)
    fixture=replace(FixturePolicy(),version='short-diagnostic-controller-1',min_support_seconds=.24)
    plan=plan_fixture(audio,22050,times,curve,filtered[1],disagreement,reliable,protected,fixture)
    return dict(curve=curve,plan=plan,predictions=filtered,protected=protected,reliable=reliable,
        candidates=candidates,policy=asdict(policy),
        uncertainty_meaning='Boundary-view disagreement only, not calibrated probability or musical intention',
        production_eligible=False)
