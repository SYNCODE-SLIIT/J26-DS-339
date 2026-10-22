"""Fixed-center D/E context diagnostic; no waveform rendering or acceptance."""
import numpy as np
from scipy.signal import butter,hilbert,sosfiltfilt
from bpdr_input.experiments.audio.segment_phase import interval_summary
from bpdr_input.preparation.diagnostics.support_fragmentation import diagnose, intervals

VERSION='fixed-center-context-diagnostic-0.1'


def analytic_parts(filtered,sr):
    z=hilbert(filtered)
    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite analytic signal')
    phase=np.unwrap(np.angle(z))
    return phase+np.pi/2,abs(z),np.gradient(phase)*sr/(2*np.pi)


def estimate_contexts(audio_float32,sample_rate_hz,frozen_arrays,saved_islands):
    """D full filter/full Hilbert; E identical full filter/island Hilbert.

    The saved local center is consumed unchanged, without center recomputation.
    Whole-clip context includes unreliable audio; output masks remain frozen.
    Phase origins are not aligned or joined across raw islands.
    """
    audio=np.asarray(audio_float32);sr=sample_rate_hz
    if audio.dtype!=np.float32 or audio.ndim!=1 or len(audio)<2 or sr!=22050 or not np.isfinite(audio).all():
        raise ValueError('Expected finite mono float32 at 22050 Hz')
    diagnose(frozen_arrays,sr)
    if len(frozen_arrays['sample_times'])!=len(audio):raise ValueError('Audio/grid mismatch')
    regions=intervals(frozen_arrays['sample_reliable'])
    if len(regions)!=len(saved_islands):raise ValueError('Saved island count changed')
    arrays={}
    for arm in ('D','E'):
        for key in ('sine_phase','amplitude','frequency_hz'):arrays[arm+'_'+key]=np.full(len(audio),np.nan)
        arrays[arm+'_computed']=np.zeros(len(audio),dtype=bool)
    for key in ('sample_reliable','guarded_support','retained_support'):arrays[key]=frozen_arrays[key].copy()
    rows={arm:[] for arm in ('D','E')}
    for region,saved in zip(regions,saved_islands):
        lo,hi=region['start_sample'],region['end_sample']
        if (lo,hi)!=(saved['start_sample'],saved['end_sample']):raise ValueError('Saved raw extent changed')
        center=saved['local_center_hz']
        for arm in rows:rows[arm].append(dict(**region,local_center_hz=center,status='refused',reason=None))
        if hi-lo<4411:
            for arm in rows:rows[arm][-1]['reason']='SHORT_RAW_ISLAND'
            continue
        if center is None or not np.isfinite(center) or not 0<.7*center<1.3*center<sr/2:
            for arm in rows:rows[arm][-1]['reason']='INVALID_SAVED_LOCAL_BAND'
            continue
        try:
            sos=butter(4,[.7*center,1.3*center],btype='bandpass',fs=sr,output='sos')
            filtered=sosfiltfilt(sos,audio.astype(np.float64),padtype='odd',padlen=27)
            if not np.isfinite(filtered).all():raise FloatingPointError('Nonfinite filtered signal')
        except (ValueError,FloatingPointError) as exc:
            for arm in rows:rows[arm][-1].update(reason='FULL_FILTER_ERROR',error_type=type(exc).__name__,error=str(exc))
            continue
        for arm,signal in [('D',filtered),('E',filtered[lo:hi])]:
            try:phase,amp,freq=analytic_parts(signal,sr)
            except (ValueError,FloatingPointError) as exc:
                rows[arm][-1].update(reason='ANALYTIC_ERROR',error_type=type(exc).__name__,error=str(exc));continue
            if arm=='D':phase,amp,freq=phase[lo:hi],amp[lo:hi],freq[lo:hi]
            for key,value in [('sine_phase',phase),('amplitude',amp),('frequency_hz',freq)]:arrays[arm+'_'+key][lo:hi]=value
            arrays[arm+'_computed'][lo:hi]=True
            rows[arm][-1].update(status='computed',nonfinite_frequency_samples=int((~np.isfinite(freq)).sum()),
                nonpositive_frequency_samples=int((np.isfinite(freq)&(freq<=0)).sum()))
    result=dict(version=VERSION,status='diagnostic_only',production_eligible=False,accepted_audio=None,arrays=arrays)
    for arm in rows:
        result[arm]=interval_summary(arrays[arm+'_frequency_hz'],arrays['retained_support'])
        result[arm]['raw_islands']=rows[arm]
    return result
