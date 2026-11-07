"""Input-only, two-sided local pulse evidence; never intended-pitch truth or permission.

Unlike the held-tone feature, this searches160-700ms events within input acoustic
regions. Each side needs three reliable observations beyond an80ms measurement
guard. A fitted event may still be intentional expression and requires learned
preservation/uncertainty checks plus waveform verification.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class EvidencePolicy:
    version:str='two-sided-short-event-evidence-1'
    minimum_width_frames:int=8
    maximum_width_frames:int=35
    anchor_frames:int=3
    guard_frames:int=4
    maximum_fraction:float=.65
    minimum_gain:float=.75
    maximum_residual_cents:float=7.
    minimum_level_cents:float=12.
    maximum_level_cents:float=35.
    region_split_cents:float=70.


def _regions(pitch,reliable):
    result=[];start=None
    for i in range(len(pitch)+1):
        inside=i<len(pitch) and reliable[i]
        if inside and start is not None:
            level=np.median(pitch[max(start,i-25):i])
            if abs(pitch[i]-level)>70.:
                result.append((start,i));start=i
        elif inside:start=i
        elif start is not None:result.append((start,i));start=None
    return result


def evidence(raw,policy=EvidencePolicy(),details=False):
    raw=np.asarray(raw,dtype=np.float32)
    if raw.ndim!=2 or raw.shape[1]!=71 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Expected finite nonempty frames x71 input')
    if policy!=EvidencePolicy():raise ValueError('Unregistered short-event evidence policy')
    n=len(raw);out=np.zeros((n,3),np.float32);fits=[];voiced=raw[:,66]>.5
    if voiced.any():
        x=np.arange(n);pitch=np.interp(x,x[voiced],raw[voiced,64])
        smooth=median_filter(pitch,size=3,mode='nearest')
        context=voiced.copy()
        changes=np.diff(np.r_[False,~voiced,False].astype(int))
        for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            if a>0 and b<n and b-a<=2:context[a:b]=True
        for a,b in _regions(smooth,context):
            # Long holds use overlapping input windows; no event is padded.
            blocks=[(a,b)] if b-a<=101 else [(left,min(left+101,b)) for left in range(a,b,50)]
            for start,stop in blocks:
                size=stop-start;margin=policy.anchor_frames+policy.guard_frames
                if size<policy.minimum_width_frames+2*margin:continue
                y=smooth[start:stop].astype(float);t=np.linspace(-1,1,size)
                base=np.column_stack([np.ones(size),t]);null_fit=np.linalg.lstsq(base,y,rcond=None)[0]
                null=float(np.sum((y-base@null_fit)**2))
                if null<1e-8:continue
                best=None
                for width in range(policy.minimum_width_frames,min(policy.maximum_width_frames,size-2*margin)+1):
                    if width>size*policy.maximum_fraction:continue
                    for left in range(margin,size-width-margin+1):
                        right=left+width
                        left_anchor=voiced[start:start+left-policy.guard_frames]
                        right_anchor=voiced[start+right+policy.guard_frames:stop]
                        if left_anchor.sum()<3 or right_anchor.sum()<3:continue
                        pulse=np.zeros(size);pulse[left:right]=1.
                        pulse=np.convolve(pulse,[.1,.2,.4,.2,.1],mode='same')
                        design=np.column_stack([base,pulse]);coef=np.linalg.lstsq(design,y,rcond=None)[0]
                        level=float(coef[2]);residual=y-design@coef
                        error=float(np.sum(residual**2));gain=1-error/null;std=float(np.std(residual))
                        if not policy.minimum_level_cents<=abs(level)<=policy.maximum_level_cents:continue
                        if gain<policy.minimum_gain or std>policy.maximum_residual_cents:continue
                        if best is None or error<best[0]:
                            best=(error,left,right,level,gain,std,pulse,int(left_anchor.sum()),int(right_anchor.sum()))
                if best is None:continue
                _,left,right,level,gain,std,pulse,la,ra=best
                feature=np.stack([-level*pulse/30,gain*pulse,std*pulse/30],axis=1).astype(np.float32)
                # Deterministic gain priority on overlapping input windows.
                take=feature[:,1]>out[start:stop,1]
                out[start:stop][take]=feature[take]
                fits.append(dict(region_start=start,region_stop=stop,pulse_start=start+left,
                    pulse_stop=start+right,level_cents=level,gain=gain,residual_cents=std,
                    reliable_left_anchor_frames=la,reliable_right_anchor_frames=ra))
    out[~voiced]=0.
    if not np.isfinite(out).all():raise ValueError('Nonfinite short-event evidence')
    return (out,dict(policy=asdict(policy),fits=fits,permission_granted=False,
                     intended_pitch_known=False)) if details else out
