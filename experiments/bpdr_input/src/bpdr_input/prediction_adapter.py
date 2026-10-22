"""Read-only prediction-to-permission preview; no production repair authorization."""
from dataclasses import asdict, dataclass
import numpy as np
from .repair_controller import FixturePolicy, check_source, decision_hash, plan_fixture, waveform_hash

FEATURE_KEYS = {'times_seconds','f0_hz','confidence','voiced_mask','pitch_change_cents','pitch_change_valid',
                'log_energy_db','onset_score','provisional_boundary_score','log_mel_db','frame_features_70',
                'boundary_features_71','native_pitch_times_seconds','native_pitch_hz','native_pitch_confidence'}
REQUIRED = {'times_seconds','confidence','voiced_mask','provisional_boundary_score','frame_features_70','boundary_features_71'}
VIEWS = ('main','earlier_40ms','later_40ms')


@dataclass(frozen=True)
class ProposalPolicy:
    version: str = 'prediction-permission-preview-0.1'
    request_threshold_cents: float = 5.
    boundary_threshold: float = .5
    guard_seconds: float = .1

    def __post_init__(self):
        if (not all(np.isfinite(v) for v in (self.request_threshold_cents,self.boundary_threshold,self.guard_seconds))
                or self.request_threshold_cents <= 0 or not 0 < self.boundary_threshold <= 1 or self.guard_seconds < 0):
            raise ValueError('Invalid provisional proposal policy')


def validate_features(features, audio, sr):
    check_source(audio,sr)
    if not REQUIRED <= set(features) or set(features)-FEATURE_KEYS:
        raise ValueError('Expected runtime acoustic arrays only; references/targets are forbidden')
    times=np.asarray(features['times_seconds'])
    expected=np.arange(len(audio)//441+1)*.02
    frame=np.asarray(features['frame_features_70']); boundary=np.asarray(features['boundary_features_71'])
    if (times.shape!=expected.shape or not np.allclose(times,expected,rtol=0,atol=1e-12)
            or frame.shape!=(len(times),70) or boundary.shape!=(len(times),71)
            or not np.isfinite(frame).all() or not np.isfinite(boundary).all()
            or not np.array_equal(frame,boundary[:,:70])):
        raise ValueError('Invalid runtime grids or 70/71 feature alignment')
    confidence=np.asarray(features['confidence']); voiced=np.asarray(features['voiced_mask'])
    score=np.asarray(features['provisional_boundary_score'])
    if (confidence.shape!=times.shape or voiced.shape!=times.shape or score.shape!=times.shape or voiced.dtype!=bool
            or not np.isfinite(confidence).all() or np.any((confidence<0)|(confidence>1))
            or not np.isfinite(score).all() or np.any((score<0)|(score>1))
            or not np.allclose(frame[:,65],confidence,rtol=0,atol=1e-6)
            or not np.array_equal(frame[:,66],voiced.astype(frame.dtype))
            or not np.allclose(boundary[:,-1],score,rtol=0,atol=1e-6)):
        raise ValueError('Acoustic fields and feature columns disagree')
    return times


def runtime_views(features):
    """Sensitivity hypotheses shift the heuristic boundary column by +/-40 ms.

    They never shift/crop audio, timestamps or the first 70 acoustic columns.
    These are not note annotations or a calibrated natural-boundary estimator.
    """
    original=np.asarray(features['boundary_features_71'])
    views={'main':original.copy()}
    for name,offset in ((VIEWS[1],-2),(VIEWS[2],2)):
        shifted=np.zeros(len(original),dtype=original.dtype)
        if len(original)>abs(offset):
            if offset<0: shifted[:offset]=original[-offset:,-1]
            else: shifted[offset:]=original[:-offset,-1]
        values=original.copy(); values[:,-1]=shifted; views[name]=values
    return views


def infer_views(model, normalizer, features):
    """Frozen eval forwards with the saved training normalizer; no refit/dropout uncertainty."""
    import torch
    if model.training:
        raise ValueError('Prediction preview requires model.eval()')
    width=model.config.features
    if width not in (70,71): raise ValueError('Only the declared 70/71-feature models are supported')
    mean,scale=np.asarray(normalizer.mean),np.asarray(normalizer.scale)
    if (mean.shape!=scale.shape or mean.ndim!=1 or len(mean)<width or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale<=0)):
        raise ValueError('Invalid saved training normalizer')
    device=next(model.parameters()).device
    result={}
    with torch.inference_mode():
        for name,values in runtime_views(features).items():
            normalized=normalizer(values[:,:width]).astype(np.float32)
            predicted=model(torch.from_numpy(normalized).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            if predicted.shape!=(len(values),) or not np.isfinite(predicted).all():
                raise ValueError('Model returned invalid correction frames')
            result[name]=predicted.copy()
    return result


def candidate_intervals(times, curve, duration):
    """Split on gaps and sign changes; never bridge a zero/rest or opposing edits."""
    requested=np.flatnonzero(curve!=0)
    if not len(requested): return []
    split=(np.diff(requested)>1)|(np.sign(curve[requested[1:]])!=np.sign(curve[requested[:-1]]))
    groups=np.split(requested,np.flatnonzero(split)+1)
    return [dict(start_seconds=float(times[g[0]]),end_seconds=min(float(times[g[-1]]+.02),duration),
                 first_frame=int(g[0]),last_frame=int(g[-1]),frames=len(g),
                 direction='raise' if curve[g[0]]>0 else 'lower',
                 maximum_abs_cents=float(np.max(abs(curve[g]))),
                 varying_curve=bool(np.ptp(curve[g])>0)) for g in groups]


def propose(audio,sr,features,predictions,uncertainty=None,expressive_mask=None,
            mode='runtime_preview',policy=ProposalPolicy(),controller_policy=FixturePolicy()):
    """Connect raw predictions to the existing fixture permission rules.

    Runtime preview never grants edits: no calibrated permission/uncertainty or
    expressivity provider exists yet. Only explicitly labelled synthetic-fixture
    mode can expose a permitted fixture plan; neither mode grants production use.
    """
    if mode not in ('runtime_preview','synthetic_fixture'): raise ValueError('Unknown adapter mode')
    times=validate_features(features,audio,sr)
    if set(predictions)!=set(VIEWS): raise ValueError('All declared boundary-view predictions are required')
    raw={name:np.asarray(predictions[name],dtype=np.float64) for name in VIEWS}
    if any(v.shape!=times.shape or not np.isfinite(v).all() for v in raw.values()):
        raise ValueError('Invalid prediction units/grid/finite values')
    # A declared provisional deadband; do not clip/smooth over-limit values.
    curve=np.where(abs(raw['main'])>=policy.request_threshold_cents,raw['main'],0.)
    alternatives=np.stack([raw[n] for n in VIEWS[1:]])
    worst=np.argmax(abs(alternatives-raw['main']),axis=0)
    alternate=alternatives[worst,np.arange(len(times))]
    reliable=np.asarray(features['voiced_mask']) & (np.asarray(features['confidence'])>=.8)
    protected=(times<=policy.guard_seconds)|(times>=len(audio)/sr-policy.guard_seconds)
    scores=np.maximum.reduce([v[:,-1] for v in runtime_views(features).values()])
    unsafe=(scores>=policy.boundary_threshold)|~reliable
    for location in times[unsafe]: protected |= abs(times-location)<=policy.guard_seconds+1e-12
    blockers=[]
    if expressive_mask is None: blockers.append('EXPRESSIVITY_PROTECTION_UNAVAILABLE')
    else:
        expressive_mask=np.asarray(expressive_mask)
        if expressive_mask.shape!=times.shape or expressive_mask.dtype!=bool: raise ValueError('Invalid expressivity mask')
        protected |= expressive_mask
    if uncertainty is None: blockers.append('UNCERTAINTY_UNAVAILABLE')
    else:
        uncertainty=np.asarray(uncertainty,dtype=np.float64)
        if uncertainty.shape!=times.shape or not np.isfinite(uncertainty).all() or np.any(uncertainty<0):
            raise ValueError('Invalid uncertainty array')
    if mode=='runtime_preview':
        blockers.extend(['UNCERTAINTY_UNCALIBRATED','PRODUCTION_PERMISSION_POLICY_UNCALIBRATED'])
    intervals=candidate_intervals(times,curve,len(audio)/sr)
    # Fail closed when a sign flip creates shorter supports than the controller permits.
    split_short=any(i['end_seconds']-i['start_seconds']<controller_policy.min_support_seconds-1e-12 for i in intervals)
    if split_short: blockers.append('SIGN_OR_GAP_SPLIT_SUPPORT_TOO_SHORT')
    plan=plan_fixture(audio,sr,times,curve,alternate,uncertainty,reliable,protected,controller_policy)
    if plan['permitted'] and intervals:
        plan['intervals_seconds']=[(i['start_seconds'],i['end_seconds']) for i in intervals]
        plan['decision_sha256']=decision_hash(plan)
    allowed=bool(mode=='synthetic_fixture' and not blockers and plan['permitted'])
    report=dict(scope='engineering_prediction_preview',mode=mode,production_eligible=False,accepted_audio=None,
        source_sha256=waveform_hash(audio),sample_rate_hz=sr,sample_count=len(audio),
        proposal_policy=asdict(policy),candidate_intervals=intervals,requested_frames=int(np.count_nonzero(curve)),
        raw_max_abs_cents=float(np.max(abs(raw['main']))),worst_boundary_disagreement_cents=float(np.max(abs(raw['main']-alternate))),
        below_deadband_frames=int(np.count_nonzero((raw['main']!=0)&(curve==0))),
        edit_permission=allowed,usable_intervals_seconds=plan['intervals_seconds'] if allowed else [],
        blockers=list(dict.fromkeys(blockers+plan['reasons'])),
        outcome='NO_EDIT_PROPOSED' if not np.any(curve) else ('FIXTURE_PROPOSAL_PERMITTED' if allowed else 'WITHHELD'),
        fixture_permission_plan=plan if allowed else None,
        meaning='Candidate intervals and no-edit output do not establish musical suitability or repair benefit')
    arrays=dict(times_seconds=times.copy(),raw_main=raw['main'],raw_earlier=raw[VIEWS[1]],raw_later=raw[VIEWS[2]],
                requested_curve=curve,reliable_mask=reliable,protected_mask=protected,
                worst_alternate=alternate)
    return report,arrays
