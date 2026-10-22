"""Experimental Praat pitch-tier renderer; never grants production edit permission."""
from dataclasses import asdict, dataclass
import numpy as np
from .interventions import envelope
from .repair_controller import check_source


@dataclass(frozen=True)
class RendererPolicy:
    version: str = 'praat-framewise-candidate-0.1'
    parselmouth_version: str = '0.4.7'
    time_step_seconds: float = .01
    pitch_floor_hz: float = 50.
    pitch_ceiling_hz: float = 1500.
    ramp_seconds: float = .08
    max_cents: float = 30.
    min_support_seconds: float = .86

    def __post_init__(self):
        values=(self.time_step_seconds,self.pitch_floor_hz,self.pitch_ceiling_hz,self.ramp_seconds,
                self.max_cents,self.min_support_seconds)
        if (not all(np.isfinite(v) and v>0 for v in values) or self.pitch_floor_hz>=self.pitch_ceiling_hz
                or self.min_support_seconds<=2*self.ramp_seconds):
            raise ValueError('Invalid candidate renderer policy')


def render_candidate(audio,sr,times,correction_cents,intervals,policy=RendererPolicy()):
    """Relative correction scales Praat's measured pitch; no target MIDI/scale snapping.

    Piecewise-linear correction interpolation is a rendering hypothesis, not a
    promise of an exact waveform/pitch ramp. Empty duration tier remains intact.
    Full resynthesis is private; samples outside the local envelope are copied.
    """
    check_source(audio,sr)
    times=np.asarray(times);correction=np.asarray(correction_cents)
    expected=np.arange(len(audio)//441+1)*.02
    if (times.shape!=expected.shape or not np.allclose(times,expected,rtol=0,atol=1e-12)
            or correction.shape!=times.shape or not np.isfinite(correction).all()
            or np.max(abs(correction))>policy.max_cents):
        raise ValueError('Invalid correction grid or over-limit curve; no clipping')
    intervals=tuple((float(a),float(b)) for a,b in intervals)
    previous=0.
    covered=np.zeros(len(times),dtype=bool)
    blend=np.zeros(len(audio))
    sample_times=np.arange(len(audio))/sr
    for start,end in intervals:
        if (not np.isfinite(start+end) or start<previous or not 0<=start<end<=len(audio)/sr
                or end-start<policy.min_support_seconds-1e-12):
            raise ValueError('Invalid, overlapping, unordered or too-short supports')
        previous=end;covered|=(times>=start)&(times<end)
        blend=np.maximum(blend,envelope(sample_times,(start,end),policy.ramp_seconds))
    if np.any((correction!=0)&~covered): raise ValueError('Correction outside permitted support')
    diagnostic=dict(scope='experimental_renderer_only',policy=asdict(policy),production_eligible=False,
                    accepted_audio=None,identity=not bool(np.any(correction)),duration_tier_modified=False,
                    same_sample_count=True,outside_samples_exact=True)
    if not np.any(correction): return audio.copy(),diagnostic
    import parselmouth
    from parselmouth.praat import call
    if parselmouth.__version__!=policy.parselmouth_version: raise ValueError('Unpinned Parselmouth version')
    sound=parselmouth.Sound(audio.astype(np.float64),sampling_frequency=sr,start_time=0.)
    manipulation=call(sound,'To Manipulation',policy.time_step_seconds,policy.pitch_floor_hz,policy.pitch_ceiling_hz)
    original_tier=call(manipulation,'Extract pitch tier')
    count=int(call(original_tier,'Get number of points'))
    if count==0: raise ValueError('No measured pitch points; no resynthesis permitted')
    tier=call('Create PitchTier','bpdr_candidate',sound.xmin,sound.xmax)
    points=[]
    for index in range(1,count+1):
        point_time=float(call(original_tier,'Get time from index',index))
        frequency=float(call(original_tier,'Get value at index',index))
        cents=float(np.interp(point_time,times,correction,left=0.,right=0.))
        shifted=frequency*2**(cents/1200.)
        if not np.isfinite(shifted) or not 0<shifted: raise ValueError('Invalid scaled pitch tier')
        call(tier,'Add point',point_time,shifted);points.append(point_time)
    if any(not any(start<=t<end for t in points) for start,end in intervals):
        raise ValueError('Support lacks measured pitch points')
    call([tier,manipulation],'Replace pitch tier')
    rendered=call(manipulation,'Get resynthesis (overlap-add)')
    if (rendered.values.shape!=(1,len(audio)) or rendered.sampling_frequency!=sr
            or rendered.xmin!=sound.xmin or rendered.xmax!=sound.xmax or rendered.x1!=sound.x1):
        raise ValueError('Resynthesis changed sample count/rate/time origin; no padding or trimming')
    full=rendered.values[0]
    candidate=(audio*(1-blend)+full*blend).astype(np.float32)
    if not np.isfinite(candidate).all() or np.max(abs(candidate))>=1:
        raise ValueError('Nonfinite/clipped candidate; no automatic normalization')
    if not np.array_equal(candidate[blend==0],audio[blend==0]): raise ValueError('Outside samples changed')
    diagnostic.update(parselmouth_version=parselmouth.__version__,praat_version=parselmouth.PRAAT_VERSION,
        native_pitch_tier_points=count,source_time_origin_seconds=sound.xmin,
        source_first_sample_seconds=sound.x1,output_time_origin_seconds=rendered.xmin,
        correction_definition='Measured Praat tier frequency multiplied by 2**(interpolated correction cents/1200)')
    return candidate,diagnostic
