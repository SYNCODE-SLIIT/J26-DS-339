"""Fixed three-harmonic source decomposition, without correction or oracle inputs."""
import numpy as np

VERSION='source-estimated-three-harmonic-separation-0.1'


def relative_rmse(estimate, reference, mask):
    estimate=np.asarray(estimate);reference=np.asarray(reference);mask=np.asarray(mask)
    if (estimate.ndim!=1 or reference.shape!=estimate.shape or mask.shape!=estimate.shape or mask.dtype!=bool
            or not mask.any() or not np.isfinite(estimate[mask]).all() or not np.isfinite(reference[mask]).all()):
        raise ValueError('Invalid reconstruction score support')
    reference_rms=float(np.sqrt(np.mean(reference[mask]**2)))
    if reference_rms<=0:raise ValueError('No reference energy; no epsilon denominator')
    rmse=float(np.sqrt(np.mean((estimate[mask]-reference[mask])**2)))
    return dict(samples=int(mask.sum()),rmse=rmse,reference_rms=reference_rms,relative_rmse=rmse/reference_rms)


def zero_reconstruction(audio, supported_model):
    """Measure raw arithmetic. An exact no-edit bypass is a separate output."""
    source=np.asarray(audio);model=np.asarray(supported_model)
    if source.dtype!=np.float32 or source.ndim!=1 or model.shape!=source.shape or not np.isfinite(model).all():
        raise ValueError('Invalid decomposition source/model')
    residual=source.astype(np.float64)-model
    raw=model+residual
    return residual,raw,raw.astype(np.float32),source.copy()


def separate_source(audio_float32, sample_rate_hz, prediction):
    """Receives frozen audio-only prediction values, never source labels/truth.

    Six sine/cosine coefficients are fitted on retained samples before the clip
    midpoint. Estimated phase was computed using the entire clip; the second
    half is excluded only from coefficient fitting, not from phase estimation.
    """
    diagnostic=dict(version=VERSION,status='withheld',reasons=[],production_eligible=False,
                    ordinary_vocal_renderer=False,accepted_audio=None,arrays={})
    if prediction['status']!='estimated':
        diagnostic['reasons']=['PRIOR_ESTIMATOR_REFUSAL'];return diagnostic
    audio=np.asarray(audio_float32);sr=sample_rate_hz;a=prediction['arrays']
    if audio.ndim!=1 or audio.dtype!=np.float32 or sr!=22050 or not np.isfinite(audio).all():
        raise ValueError('Expected finite mono float32 source at 22050 Hz')
    t=np.arange(len(audio))/sr;phase=np.asarray(a['estimated_sine_phase']);support=np.asarray(a['retained_support'])
    frequency=np.asarray(a['estimated_frequency_hz'])
    if (not np.array_equal(a['sample_times'],t) or phase.shape!=t.shape or support.shape!=t.shape
            or support.dtype!=bool or frequency.shape!=t.shape or not np.isfinite(phase).all()
            or not np.isfinite(frequency).all() or np.any(frequency<=0)):
        raise ValueError('Changed/nonpositive source-only phase/frequency/support')
    fit=support&(t<len(audio)/(2*sr));evaluation=support&~fit
    if fit.sum()<6 or not evaluation.any():
        diagnostic.update(status='refused',reasons=['INSUFFICIENT_FIT_OR_EVALUATION_SUPPORT']);return diagnostic
    basis=np.column_stack([fn(k*phase) for k in (1,2,3) for fn in (np.sin,np.cos)])
    coefficients,_,rank,singular=np.linalg.lstsq(basis[fit],audio[fit].astype(np.float64),rcond=None)
    if rank!=6 or not np.isfinite(coefficients).all():
        diagnostic.update(status='refused',reasons=['RANK_DEFICIENT_OR_NONFINITE_FIT']);return diagnostic
    partials=np.column_stack([basis[:,2*k:2*k+2]@coefficients[2*k:2*k+2] for k in range(3)])
    partials[~support]=0.;model=partials.sum(axis=1)
    residual,raw,cast,identity=zero_reconstruction(audio,model)
    diagnostic.update(status='decomposed',reasons=[],coefficient_order=['sin(phi)','cos(phi)','sin(2phi)','cos(2phi)','sin(3phi)','cos(3phi)'],
        coefficients=coefficients.tolist(),rank=int(rank),condition_number=float(singular[0]/singular[-1]),
        fit_samples=int(fit.sum()),evaluation_samples=int(evaluation.sum()),supported_samples=int(support.sum()),
        excluded_samples=int((~support).sum()),
        harmonic_source_fit=relative_rmse(model,audio.astype(float),fit),
        harmonic_source_evaluation=relative_rmse(model,audio.astype(float),evaluation),
        raw_recombination_max_abs_error=float(np.max(abs(raw-audio.astype(float)))),
        raw_recombination_float32_different_samples=int(np.count_nonzero(cast!=audio)),
        raw_recombination_outside_exact=bool(np.array_equal(cast[~support],audio[~support])),
        zero_edit_bypass_exact=bool(np.array_equal(identity,audio)),sample_count_unchanged=raw.shape==audio.shape,
        arrays=dict(sample_times=t,estimated_sine_phase=phase,retained_support=support,fit_support=fit,evaluation_support=evaluation,
                    estimated_partials=partials,supported_harmonic_model=model,residual=residual,
                    raw_recombination=raw,raw_recombination_float32=cast,zero_edit_bypass=identity))
    return diagnostic
