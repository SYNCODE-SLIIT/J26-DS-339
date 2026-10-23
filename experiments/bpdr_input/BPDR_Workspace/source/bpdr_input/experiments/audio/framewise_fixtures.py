"""Declared controlled framewise rendering experiments; no adoption or real-vocal edits."""
from bpdr_input import COMPONENT_ROOT, module_source
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.audio.analysis import extract
from bpdr_input.experiments.audio.audit_diagnostics import pitch_metrics
from bpdr_input.preparation.sources.data_builder import write_json
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.framewise_renderer import RendererPolicy, render_candidate
from bpdr_input.repair.interventions import AuditPolicy, independent_pitch
from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair.prediction_adapter import VIEWS, propose
from bpdr_input.repair.repair_controller import transact_fixture, waveform_hash

PROFILES=[dict(name=f'constant_{hz}',hz=hz,kind='constant',seconds=4) for hz in (110,220,440,880)]
PROFILES += [dict(name=f'vibrato_{hz}',hz=hz,kind='vibrato',seconds=4) for hz in (220,880)]
PROFILES += [dict(name='glide_220',hz=220,kind='glide',seconds=4),dict(name='rested_220',hz=220,kind='rest',seconds=8)]
CURVES=('minus20','plus20','rising10to30','crossing_minus30to30')


def constructed(profile,sr=22050):
    t=np.arange(profile['seconds']*sr)/sr
    contour=35.*np.sin(2*np.pi*5*t) if profile['kind']=='vibrato' else (-100.+50.*t if profile['kind']=='glide' else np.zeros(len(t)))
    f0=profile['hz']*2**(contour/1200.)
    amplitude=np.full(len(t),.1)
    if profile['kind']=='rest': amplitude[(t>=2.5)&(t<4.5)]=0.
    return t,f0,amplitude


def signal(f0,amplitude,sr=22050):
    phase=2*np.pi*np.cumsum(f0)/sr
    return (amplitude*(np.sin(phase)+.4*np.sin(2*phase)+.2*np.sin(3*phase))).astype(np.float32)


def requested_curve(times,intervals,name):
    values=np.zeros(len(times))
    for start,end in intervals:
        mask=(times>=start)&(times<end);u=(times[mask]-start)/(end-start)
        values[mask]=(-20. if name=='minus20' else 20. if name=='plus20' else 10.+20.*u if name=='rising10to30' else -30.+60.*u)
    return values


def run():
    import parselmouth
    import torch
    component=COMPONENT_ROOT
    root=component/'runs/framewise_fixtures'/uuid.uuid4().hex;root.mkdir(parents=True)
    snapshots=root/'source_code_at_run';snapshots.mkdir()
    code={}
    for name in ('framewise_fixtures','framewise_renderer','framewise_audit','analysis','interventions','prediction_adapter','repair_controller'):
        path=module_source(name+'.py');code[name]=checksum(path);(snapshots/path.name).write_bytes(path.read_bytes())
    record=dict(run_status='running',scope='experimental_framewise_renderer',data_status='exploratory_not_research',
        renderer_policy=asdict(RendererPolicy()),base_audit_limits=asdict(AuditPolicy()),
        profiles=PROFILES,declared_curves=CURVES,parselmouth_version=parselmouth.__version__,praat_version=parselmouth.PRAAT_VERSION,
        backend_path=parselmouth.__file__,backend_sha256=checksum(parselmouth.__file__),cpu_threads=torch.get_num_threads(),
        code_sha256=code,artifacts=[],cases=[],identity_cases=[],adopted=False,accepted_prepared_packages=0,training_ready_examples=0,
        limitations=['Constructed signals only; no real vocal accuracy or audible-quality ratings',
            'Per-frame candidate audit retains v0.3 magnitude/coverage/drift limits and adds stricter framewise direction at >=10 cents',
            'No tuning after outcomes, no active lab renderer/policy change, and no production permission',
            'Analytic oracle is an estimator diagnostic; its outside waveform phase is not an admissible local renderer',
            'Prototype controller integration uses supplied curve/uncertainty/expressivity fixtures, not validated model proposals'])
    def save():write_json(root/'evidence.json',record)
    def artifact(path):record['artifacts'].append(dict(path=str(path),sha256=checksum(path)))
    def save_wave(path,wave):
        sf.write(path,wave,22050,subtype='FLOAT');loaded,rate=sf.read(path,dtype='float32')
        if rate!=22050 or not np.array_equal(loaded,wave):raise ValueError('Saved candidate changed samples')
        artifact(path)
    save();print('FRAMEWISE_FIXTURE_RUN:',root,flush=True)
    try:
        for profile in PROFILES:
            print('PROFILE:',profile['name'],flush=True)
            sample_times,f0,amplitude=constructed(profile);original=signal(f0,amplitude)
            before,analyzer=extract(original,22050,component/'checkpoints');yin=independent_pitch(original,22050)
            times=before['times_seconds'];reliable=before['voiced_mask']&np.isfinite(yin)&(yin>0)
            reliable[reliable]&=abs(1200*np.log2(yin[reliable]/before['f0_hz'][reliable]))<=AuditPolicy().tracker_agreement_cents
            if profile['kind']=='rest':reliable&=(times<2.5)|(times>=4.5)
            intervals=[(1.,2.)] if profile['seconds']==4 else [(1.,2.),(5.,6.)]
            folder=root/profile['name'];folder.mkdir();save_wave(folder/'source.wav',original)
            identity,identity_diagnostic=render_candidate(original,22050,times,np.zeros(len(times)),[])
            if not np.array_equal(identity,original):raise ValueError('Identity failed')
            record['identity_cases'].append(dict(profile=profile['name'],exact=True))
            for name in CURVES:
                curve=requested_curve(times,intervals,name)
                candidate,diagnostic=render_candidate(original,22050,times,curve,intervals)
                after,_=extract(candidate,22050,component/'checkpoints')
                measured=audit_curve(before,after,yin,independent_pitch(candidate,22050),reliable,intervals,curve)
                # A known contour separates estimator/reference limitations from renderer failure.
                oracle=signal(f0*2**(np.interp(sample_times,times,curve)/1200.),amplitude)
                oracle_yin=independent_pitch(oracle,22050)
                mask=np.zeros(len(times),dtype=bool)
                for start,end in intervals:mask|=(times>=start+.18)&(times<=end-.18)
                oracle_delta=1200*np.log2(oracle_yin/yin)
                errors=abs(oracle_delta[mask]-curve[mask])
                oracle_metrics=dict(frames=int(mask.sum()),median_error_cents=float(np.median(errors)),p90_error_cents=float(np.quantile(errors,.9)))
                save_wave(folder/(name+'.wav'),candidate)
                record['cases'].append(dict(profile=profile['name'],curve=name,intervals=intervals,renderer=diagnostic,audit=measured,
                    oracle_fine=oracle_metrics,source_analyzer=analyzer,candidate_path=str(folder/(name+'.wav')),
                    candidate_waveform_sha256=waveform_hash(candidate),source_path=str(folder/'source.wav')))
                save()
        # Connect a supplied changing curve through adapter -> controller -> renderer -> audit.
        profile=dict(hz=220,kind='constant',seconds=8)
        st,f0,amp=constructed(profile);original=signal(f0,amp)
        before,_=extract(original,22050,component/'checkpoints');times=before['times_seconds'];yin=independent_pitch(original,22050)
        curve=np.where((times>=1.)&(times<1.9),10.+10.*np.clip((times-1.)/.9,0,1),0.)
        proposal,arrays=propose(original,22050,before,{v:curve.copy() for v in VIEWS},
            uncertainty=np.ones(len(times)),expressive_mask=np.zeros(len(times),dtype=bool),mode='synthetic_fixture')
        record['integration']=dict(adapter=proposal)
        if proposal['edit_permission']:
            reliable=before['voiced_mask']&np.isfinite(yin)&(yin>0)
            reliable[reliable]&=abs(1200*np.log2(yin[reliable]/before['f0_hz'][reliable]))<=50.
            def renderer(x,rate,support):return render_candidate(x,rate,times,arrays['requested_curve'],support)[0]
            def auditor(x,candidate,rate,support):
                after,_=extract(candidate,rate,component/'checkpoints')
                return audit_curve(before,after,yin,independent_pitch(candidate,rate),reliable,support,arrays['requested_curve'])
            output,result=transact_fixture(original,22050,proposal['fixture_permission_plan'],renderer,auditor)
            record['integration'].update(transaction=result,returned_source_unchanged=bool(np.array_equal(output,original)))
            save_wave(root/'integration_source.wav',original)
            save_wave(root/'integration_output_fixture_only.wav',output)
        else:record['integration']['note']='Adapter withheld the supplied fixture; no renderer invoked'
        summary={}
        for name in CURVES:
            selected=[c for c in record['cases'] if c['curve']==name]
            summary[name]=dict(tested=len(selected),audit_passes=sum(c['audit']['passed'] for c in selected),
                oracle_fine_passes=sum(c['oracle_fine']['median_error_cents']<=5. and c['oracle_fine']['p90_error_cents']<=10. for c in selected))
        record['summary']=summary
        if any(checksum(module_source(n+'.py'))!=h for n,h in code.items()):raise ValueError('Source changed during experiment')
        if checksum(parselmouth.__file__)!=record['backend_sha256']:raise ValueError('Backend changed')
        record['run_status']='complete';save()
        print(json.dumps(dict(summary=summary,integration=record['integration'].get('transaction',{}).get('outcome')),indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
