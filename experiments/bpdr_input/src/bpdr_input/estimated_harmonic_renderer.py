"""Experimental strict local phase edits from frozen audio-only estimates."""
import numpy as np
from .anchor_feasibility import solve_support
from .framewise_renderer import RendererPolicy
from .repair_controller import check_source

VERSION='estimated-harmonic-local-candidate-0.1'


def render_estimated_candidate(audio,sr,prediction,decomposition,times,requested,intervals):
    """No profile/known F0/phase/partial coefficients are accepted by this API.

    The existing numerical closure kernel is given estimated frequencies only.
    The residual stays unchanged; no relaxed solution or endpoint replacement is
    used. A candidate is experimental audio, never an accepted delivery package.
    """
    check_source(audio,sr);policy=RendererPolicy()
    diagnostic=dict(version=VERSION,outcome='WITHHELD',reasons=[],production_eligible=False,
                    accepted_audio=None,oracle_inputs_used=False,identity=False,supports=[])
    arrays={}
    if prediction['status']!='estimated' or decomposition['status']!='decomposed':
        diagnostic['reasons']=['PRIOR_SOURCE_REFUSAL'];return None,diagnostic,arrays
    times=np.asarray(times);requested=np.asarray(requested);st=np.arange(len(audio))/sr
    expected=np.arange(len(audio)//441+1)*.02
    if (times.shape!=expected.shape or not np.allclose(times,expected,rtol=0,atol=1e-12) or requested.shape!=times.shape
            or not np.isfinite(requested).all() or abs(requested).max()>policy.max_cents):
        raise ValueError('Changed frame grid or invalid/over-limit request')
    p=prediction['arrays'];d=decomposition['arrays'];support=np.asarray(p['retained_support'])
    phase=np.asarray(p['estimated_sine_phase']);frequency=np.asarray(p['estimated_frequency_hz'])
    coefficients=np.asarray(decomposition['coefficients']);residual=np.asarray(d['residual'])
    if (not np.array_equal(p['sample_times'],st) or not np.array_equal(d['sample_times'],st)
            or not np.array_equal(phase,d['estimated_sine_phase']) or not np.array_equal(support,d['retained_support'])
            or support.shape!=st.shape or support.dtype!=bool or phase.shape!=st.shape or frequency.shape!=st.shape
            or residual.shape!=st.shape or coefficients.shape!=(6,) or not np.isfinite(phase).all()
            or not np.isfinite(frequency).all() or np.any(frequency<=0) or not np.isfinite(coefficients).all()
            or not np.isfinite(residual).all()):raise ValueError('Changed frozen source estimates/decomposition')
    model=np.asarray(d['supported_harmonic_model'])
    if (model.shape!=st.shape or not np.array_equal(audio.astype(float)-model,residual)
            or not np.array_equal((model+residual).astype(np.float32),audio)):
        raise ValueError('Source/decomposition identity mismatch')
    intervals=tuple((float(a),float(b)) for a,b in intervals);previous=0.
    covered=np.zeros(len(times),dtype=bool);inside=np.zeros(len(audio),dtype=bool);masks=[]
    for start,end in intervals:
        if (not np.isfinite(start+end) or start<previous or not 0<=start<end<=len(audio)/sr
                or end-start<policy.min_support_seconds-1e-12):raise ValueError('Invalid local supports')
        previous=end;covered|=(times>=start)&(times<end);mask=(st>=start)&(st<end);inside|=mask;masks.append(mask)
    if np.any((requested!=0)&~covered):raise ValueError('Request outside local supports')
    if not requested.any():
        diagnostic.update(outcome='IDENTITY',identity=True,reasons=[]);return audio.copy(),diagnostic,arrays
    if np.any(inside&~support):
        diagnostic['reasons']=['UNSUPPORTED_SOURCE_SAMPLES'];return None,diagnostic,arrays
    sample_requested=np.interp(st,times,requested);solutions=[]
    for number,((start,end),mask) in enumerate(zip(intervals,masks)):
        result,a=solve_support(frequency[mask],sample_requested[mask],sr,round(.08*sr))
        strict=result['sign_preserving']
        diagnostic['supports'].append(dict(start_seconds=start,end_seconds=end,strict=strict))
        arrays.update({f's{number}_'+k:v for k,v in a.items() if not k.startswith('relaxed')})
        arrays[f's{number}_times_seconds']=st[mask]
        if strict['phase_closed_solution']:solutions.append(a)
    if len(solutions)!=len(masks):
        diagnostic['reasons']=['NO_STRICT_PHASE_SOLUTION'];return None,diagnostic,arrays
    output=audio.copy();delta_phase=np.zeros(len(audio));target=np.zeros(len(audio));edited_model=model.copy()
    for number,(mask,a) in enumerate(zip(masks,solutions)):
        cents=a['sign_preserving_cents'];cycles=a['sign_preserving_added_cycles']
        if (not np.isfinite(cents).all() or not np.isfinite(cycles).all()
                or cents[0]!=0 or cents[-1]!=0 or abs(cycles[-1]-round(float(cycles[-1])))>1e-10):
            raise ValueError('Invalid closed strict solution')
        shifted_phase=phase[mask]+2*np.pi*cycles
        parts=np.column_stack([coefficients[2*k]*np.sin((k+1)*shifted_phase)+coefficients[2*k+1]*np.cos((k+1)*shifted_phase) for k in range(3)])
        local_model=parts.sum(axis=1);local=(local_model+residual[mask]).astype(np.float32)
        if local[0]!=audio[mask][0] or local[-1]!=audio[mask][-1]:
            diagnostic['reasons']=['ENDPOINT_SAMPLE_MISMATCH'];return None,diagnostic,arrays
        output[mask]=local;edited_model[mask]=local_model;delta_phase[mask]=2*np.pi*cycles;target[mask]=cents
    if not np.isfinite(output).all() or abs(output).max()>=1:
        diagnostic['reasons']=['NONFINITE_OR_CLIPPED_OUTPUT'];return None,diagnostic,arrays
    if not np.array_equal(output[~inside],audio[~inside]):raise ValueError('Outside waveform changed')
    diagnostic.update(outcome='EXPERIMENTAL_CANDIDATE',reasons=[],sample_count_unchanged=output.shape==audio.shape,
        outside_exact=True,endpoints_exact=True,residual_unchanged=True,
        residual_meaning='The exact same float64 residual array is added; post-quantization inferred residual need not be byte-identical')
    arrays.update(sample_times=st,inside_support=inside,estimated_model_target_cents=target,
        added_phase_radians=delta_phase,edited_harmonic_model=edited_model)
    return output,diagnostic,arrays
