"""Input-only acoustic step evidence; not a note/key or intended-melody label.

Fit a local plateau plus linear performance drift within a measured pitch
region. The fitted level is a FEATURE, not authorization to render audio.
Missing observations can supply context across <=100ms only; edit permission
continues to prohibit unreliable samples and short supports.
"""
import numpy as np
from scipy.ndimage import median_filter
from functools import lru_cache


def evidence(raw):
    raw=np.asarray(raw,dtype=np.float32)
    if raw.ndim!=2 or raw.shape[1]!=71 or not np.isfinite(raw).all():raise ValueError('Invalid acoustic input')
    return _cached(np.ascontiguousarray(raw[:,[64,66]]).tobytes(),len(raw))


@lru_cache(maxsize=512)
def _cached(payload,n):
    values=np.frombuffer(payload,dtype=np.float32).reshape(n,2)
    raw=np.zeros((n,71),np.float32);raw[:,[64,66]]=values
    result=_compute_evidence(raw);result.setflags(write=False)
    return result


def _compute_evidence(raw):
    raw=np.asarray(raw)
    if raw.ndim!=2 or raw.shape[1]!=71 or not np.isfinite(raw).all():raise ValueError('Invalid acoustic input')
    n=len(raw);result=np.zeros((n,3),np.float32);voiced=raw[:,66]>.5
    if voiced.sum()<30:return result
    indices=np.arange(n);pitch=np.interp(indices,indices[voiced],raw[voiced,64])
    # Context-only gap bridging, never an edit mask.
    context=voiced.copy()
    d=np.diff(np.r_[False,~voiced,False].astype(int))
    for a,b in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)):
        if a>0 and b<n and b-a<=5:context[a:b]=True
    smooth=median_filter(pitch,size=9,mode='nearest')
    # Large acoustic changes and silence delimit fitting regions. The rolling
    # level also detects gradual transitions between distinct held pitches.
    regions=[];start=None
    for i in range(n+1):
        inside=i<n and context[i]
        if inside and start is not None:
            level=np.median(smooth[max(start,i-25):i])
            if abs(smooth[i]-level)>60.:
                regions.append((start,i));start=i
        elif inside:start=i
        elif start is not None:regions.append((start,i));start=None
    for start,stop in regions:
        # Avoid voice-onset/offset pitch settling in contextual anchors.
        a,b=start+3,stop-3
        size=b-a
        if size<63:continue
        y=smooth[a:b].astype(float);t=np.linspace(-1,1,size)
        design=np.stack([np.ones(size),t],axis=1)
        base=np.linalg.lstsq(design,y,rcond=None)[0]
        null=float(np.sum((y-design@base)**2))
        if null<1e-8:continue
        best=None
        # Real supports only; minimum duration is never created by padding.
        for width in range(43,min(100,size-20)+1,4):
            if width>size*.70:continue
            for left in range(10,size-width-9,2):
                pulse=np.zeros(size);pulse[left:left+width]=1.
                # Smooth four frames at each edge to represent a physical ramp.
                ramp=.5-.5*np.cos(np.linspace(0,np.pi,5))
                pulse[left:left+5]=ramp;pulse[left+width-5:left+width]=ramp[::-1]
                x=np.column_stack([design,pulse]);coef=np.linalg.lstsq(x,y,rcond=None)[0]
                level=float(coef[2]);residual=y-x@coef
                error=float(np.sum(residual**2));gain=1-error/null
                if not 8.<=abs(level)<=35. or gain<.65 or np.std(residual)>7.:continue
                if best is None or error<best[0]:best=(error,left,width,level,gain,pulse)
        if best is None:continue
        _,left,width,level,gain,pulse=best
        # A bounded contextual measurement may exceed permission; do not clip.
        result[a:b,0]=-level*pulse/30.
        result[a:b,1]=gain*pulse
        result[a:b,2]=np.std(y-(design@base))/30.*pulse
    return result
