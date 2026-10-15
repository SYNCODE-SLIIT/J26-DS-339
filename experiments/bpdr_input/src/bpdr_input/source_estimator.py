"""Fixed audio-only fundamental diagnostic; no construction metadata or renderer."""
from pathlib import Path
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from .analysis import extract
from .interventions import independent_pitch

VERSION = 'audio-only-source-estimator-0.1'


def nearest_indices(frame_times, sample_times):
    frames=np.asarray(frame_times);samples=np.asarray(sample_times)
    if (frames.ndim!=1 or not len(frames) or samples.ndim!=1 or not np.isfinite(frames).all()
            or not np.isfinite(samples).all() or np.any(np.diff(frames)<=0)):
        raise ValueError('Invalid estimator time grid')
    current=np.clip(np.searchsorted(frames,samples),0,len(frames)-1);previous=np.maximum(current-1,0)
    return np.where(abs(frames[previous]-samples)<abs(frames[current]-samples),previous,current)


def guarded_support(reliable_samples, sr):
    reliable=np.asarray(reliable_samples)
    if reliable.ndim!=1 or reliable.dtype!=bool or sr!=22050:raise ValueError('Invalid source support')
    guard=round(.1*sr);indices=np.arange(len(reliable))
    retained=reliable&(indices>=guard)&(indices<len(reliable)-guard)
    for j in np.flatnonzero(reliable[1:]!=reliable[:-1])+1:
        retained[max(0,j-guard):min(len(reliable),j+guard)]=False
    return retained


def estimate_source(audio_float32, sample_rate_hz):
    """Only decoded samples and rate enter this API. No oracle offset alignment.

    The declared nonpositive-frequency refusal applies to the entire computed
    frequency array, including outside retained support. Counts distinguish those
    regions; this stage does not silently weaken that refusal to retain a source.
    """
    audio=np.asarray(audio_float32);sr=sample_rate_hz
    if (audio.ndim!=1 or audio.dtype!=np.float32 or len(audio)<2 or sr!=22050
            or not np.isfinite(audio).all()):raise ValueError('Expected finite mono float32 at 22050 Hz')
    arrays={}
    record=dict(version=VERSION,status='refused',reasons=[],ordinary_vocal_renderer=False,
                production_eligible=False,accepted_audio=None,arrays=arrays)
    before,analyzer=extract(audio,sr,Path(__file__).resolve().parents[2]/'checkpoints')
    y0=independent_pitch(audio,sr);t=before['times_seconds']
    if y0.shape!=t.shape:raise ValueError('Source trackers have different grids')
    f=before['f0_hz'];confidence=before['confidence']
    reliable=(confidence>=.8)&np.isfinite(f)&(f>0)&np.isfinite(y0)&(y0>0)
    reliable[reliable]&=abs(1200*np.log2(y0[reliable]/f[reliable]))<=50.
    arrays.update(frame_times=t,frame_pesto_hz=f,frame_confidence=confidence,frame_yin_hz=y0,frame_reliable=reliable)
    record.update(analyzer=analyzer,reliable_frames=int(reliable.sum()),total_frames=len(t))
    if not reliable.any():record['reasons']=['NO_RELIABLE_SOURCE_FRAMES'];return record
    center=float(np.median(y0[reliable]));record['estimated_center_hz']=center
    if not 0<.7*center<1.3*center<sr/2:
        record['reasons']=['INVALID_ESTIMATED_BAND'];return record
    sos=butter(4,[.7*center,1.3*center],btype='bandpass',fs=sr,output='sos')
    analytic=hilbert(sosfiltfilt(sos,audio.astype(np.float64)))
    if not np.isfinite(analytic).all():record['reasons']=['NONFINITE_ANALYTIC_SIGNAL'];return record
    phase=np.unwrap(np.angle(analytic));amplitude=abs(analytic)
    frequency=np.gradient(phase)*sr/(2*np.pi)
    st=np.arange(len(audio))/sr
    indices=nearest_indices(t,st);sample_reliable=reliable[indices]
    guarded=guarded_support(sample_reliable,sr)
    arrays.update(sample_times=st,estimated_sine_phase=phase+np.pi/2,estimated_amplitude=amplitude,
                  estimated_frequency_hz=frequency,sample_reliable=sample_reliable,guarded_support=guarded)
    if not guarded.any():
        arrays['retained_support']=np.zeros(len(audio),dtype=bool)
        record['reasons']=['EMPTY_GUARDED_SUPPORT'];return record
    floor=.1*float(np.median(amplitude[guarded]));retained=guarded&(amplitude>floor)
    arrays['retained_support']=retained
    nonpositive=~np.isfinite(frequency)|(frequency<=0)
    record.update(amplitude_floor=floor,predicted_reliable_samples=int(sample_reliable.sum()),
                  guarded_samples=int(guarded.sum()),retained_samples=int(retained.sum()),
                  nonpositive_frequency_samples=int(nonpositive.sum()),
                  nonpositive_frequency_on_retained_samples=int((nonpositive&retained).sum()),
                  nonpositive_frequency_outside_retained_samples=int((nonpositive&~retained).sum()))
    reasons=[]
    if not retained.any():reasons.append('EMPTY_RETAINED_SUPPORT')
    if nonpositive.any():reasons.append('NONPOSITIVE_ESTIMATED_FREQUENCY')
    record['reasons']=reasons;record['status']='refused' if reasons else 'estimated'
    return record
