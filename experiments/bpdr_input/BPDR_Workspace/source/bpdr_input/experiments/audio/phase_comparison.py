"""Fixed 32-case polar joining experiment, with unchanged audits and diagnostics."""
from bpdr_input import COMPONENT_ROOT, module_source
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.audio.analysis import extract
from bpdr_input.experiments.audio.boundary_comparison import advancement
from bpdr_input.preparation.sources.data_builder import write_json
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.experiments.audio.framewise_diagnostics import relative_peak, transition_mask
from bpdr_input.experiments.audio.framewise_fixtures import requested_curve
from bpdr_input.repair.framewise_renderer import RendererPolicy
from bpdr_input.repair.interventions import AuditPolicy, independent_pitch
from bpdr_input.experiments.audio.phase_feasibility import analytic_fundamental, mixing_arrays, ramp_metrics
from bpdr_input.experiments.audio.phase_renderer import VERSION, render_phase_candidate
from bpdr_input.audio.pipeline import checksum

BASELINE='runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/evidence.json'
BOUNDARY='runs/boundary_comparison/e69c9a73c3f54f7c843701f5dadbf522/evidence.json'
PHASE='runs/phase_feasibility/d215daf94aa749ce8798f72ab13976d9/evidence.json'
DECLARATION=dict(gate='Gate F — controlled phase-aware joining candidate',
    selection='All eight existing constructed profiles x four curves; same source WAVs and supports',
    method='Original raw-curve PSOLA branch; full-band Hilbert amplitude/relative-phase interpolation only inside existing 80-ms waveform ramps',
    phase_branch='Relative phase unwrapped separately per ramp; integer cycle chosen to put the sample nearest half blend in [-pi,pi)',
    endpoints='Original samples copied exactly at blend zero; original resynthesized samples copied exactly at blend one',
    phase_amplitude_floor=1e-8,floor_meaning='Fixed numerical refusal floor, not calibrated musical suitability',
    hypothesis='Polar joining may reduce amplitude cancellation but can introduce phase-derivative pitch excursions or harmonic distortion',
    audit='Unchanged framewise-render-audit-0.1 against original requested interior curve; no epsilon or target change',
    diagnostics='Both fixed YIN windows (2048/1024); same known-center fundamental phase/amplitude diagnostic on all 72 ramps',
    advancement_filter='Reuse the frozen prior engineering filter: physical/identity/fine checks, no lost baseline pass, no count/max transition regression at either window and some transition-summary improvement',
    advancement_meaning='Considering a bounded engineering vocal comparison only; never automatic audio acceptance or adoption',
    tuning='One fixed candidate, no case exclusions, parameter search or outcome-dependent retuning',
    musical_review='Deferred by owner until experienced reviewers become available; engineering continues',
    adopted=False,accepted_prepared_packages=0,training_ready_examples=0)


def run():
    import librosa
    import scipy
    import parselmouth
    import torch
    component=COMPONENT_ROOT
    baseline=json.loads((component/BASELINE).read_text(encoding='utf-8'))
    boundary=json.loads((component/BOUNDARY).read_text(encoding='utf-8'))
    phase=json.loads((component/PHASE).read_text(encoding='utf-8'))
    if any(v['run_status']!='complete' or v['adopted'] for v in (baseline,boundary,phase)):raise ValueError('Invalid prior evidence')
    if baseline['renderer_policy']!=asdict(RendererPolicy()) or baseline['base_audit_limits']!=json.loads(json.dumps(asdict(AuditPolicy()))):raise ValueError('Base policy changed')
    tracked=dict(phase['tracked_inputs']);tracked[str(component/PHASE)]=checksum(component/PHASE)
    for item in phase['artifacts']:tracked[item['path']]=item['sha256']
    for name in ('phase_renderer','phase_comparison'):
        path=module_source(name+'.py');tracked[str(path)]=checksum(path)
    for path,digest in tracked.items():
        if checksum(path)!=digest:raise ValueError('Changed prior input: '+path)
    if checksum(parselmouth.__file__)!=phase['backend_sha256']:raise ValueError('Native backend changed')
    root=component/'runs/phase_comparison'/uuid.uuid4().hex;root.mkdir(parents=True)
    snapshots=root/'source_code_at_run';snapshots.mkdir()
    for path in tracked:
        if Path(path).suffix=='.py':(snapshots/Path(path).name).write_bytes(Path(path).read_bytes())
    write_json(root/'declaration.json',DECLARATION)
    record=dict(run_status='running',data_status='exploratory_not_research',version=VERSION,declaration=DECLARATION,
        declaration_sha256=checksum(root/'declaration.json'),tracked_inputs=tracked,
        backend_path=parselmouth.__file__,backend_sha256=phase['backend_sha256'],
        numpy_version=np.__version__,scipy_version=scipy.__version__,parselmouth_version=parselmouth.__version__,
        praat_version=parselmouth.PRAAT_VERSION,cpu_threads=torch.get_num_threads(),cases=[],identities=[],artifacts=[],
        adopted=False,accepted_prepared_packages=0,training_ready_examples=0)
    def save():write_json(root/'evidence.json',record)
    def artifact(path):record['artifacts'].append(dict(path=str(path),sha256=checksum(path)))
    save();print('PHASE_COMPARISON_RUN:',root,flush=True)
    try:
        for profile in baseline['profiles']:
            print('PROFILE:',profile['name'],flush=True)
            cases=[c for c in baseline['cases'] if c['profile']==profile['name']]
            audio,sr=sf.read(cases[0]['source_path'],dtype='float32')
            if sr!=22050:raise ValueError('Source rate changed')
            before,analyzer=extract(audio,sr,component/'checkpoints')
            if analyzer!=cases[0]['source_analyzer']:raise ValueError('Analyzer/checkpoint changed')
            times=before['times_seconds'];st=np.arange(len(audio))/sr;yin=independent_pitch(audio,sr)
            reliable=before['voiced_mask']&np.isfinite(yin)&(yin>0)
            reliable[reliable]&=abs(1200*np.log2(yin[reliable]/before['f0_hz'][reliable]))<=AuditPolicy().tracker_agreement_cents
            if profile['kind']=='rest':reliable&=(times<2.5)|(times>=4.5)
            identity,identity_diag,_=render_phase_candidate(audio,sr,times,np.zeros(len(times)),[])
            if not np.array_equal(identity,audio) or 'praat_version' in identity_diag:raise ValueError('Identity backend bypass failed')
            record['identities'].append(dict(profile=profile['name'],exact=True,backend_bypassed=True))
            refs={2048:yin,1024:librosa.yin(audio,sr=sr,fmin=50.,fmax=1500.,frame_length=1024,hop_length=441)}
            zs=analytic_fundamental(audio,sr,profile['hz'])
            voiced=(st<2.5)|(st>=4.5) if profile['kind']=='rest' else np.ones(len(st),dtype=bool)
            floor=.1*float(np.median(abs(zs[voiced])))
            folder=root/profile['name'];folder.mkdir()
            for case in cases:
                curve=requested_curve(times,case['intervals'],case['curve'])
                candidate,renderer,branches=render_phase_candidate(audio,sr,times,curve,case['intervals'])
                old_audio,rate=sf.read(case['candidate_path'],dtype='float32')
                if rate!=sr or not np.array_equal(branches['baseline_local'],old_audio):raise ValueError('Saved baseline not reproduced')
                blend=branches['blend']
                preservation=dict(sample_count=candidate.shape==audio.shape,
                    time_origin=renderer['source_time_origin_seconds']==renderer['output_time_origin_seconds'],
                    duration_tier_unchanged=not renderer['duration_tier_modified'],
                    outside_exact=bool(np.array_equal(candidate[blend==0],audio[blend==0])),
                    interior_exact=bool(np.array_equal(candidate[blend==1],old_audio[blend==1])),
                    rests_exact=bool(np.array_equal(candidate[(st>=2.5)&(st<4.5)],audio[(st>=2.5)&(st<4.5)])) if profile['kind']=='rest' else True,
                    finite_unclipped=bool(np.isfinite(candidate).all() and abs(candidate).max()<1))
                if not all(preservation.values()):raise ValueError('Physical/interior preservation failed')
                after,_=extract(candidate,sr,component/'checkpoints');candidate_yin=independent_pitch(candidate,sr)
                audit=audit_curve(before,after,yin,candidate_yin,reliable,case['intervals'],curve)
                mask=transition_mask(times,case['intervals']);peaks={};arrays=dict(times=times,requested=curve,transition_mask=mask)
                for w in refs:
                    measured=candidate_yin if w==2048 else librosa.yin(candidate,sr=sr,fmin=50.,fmax=1500.,frame_length=w,hop_length=441)
                    peaks[str(w)],arrays[f'transition_delta_{w}']=relative_peak(refs[w],measured,mask)
                mix=mixing_arrays(zs,analytic_fundamental(branches['full'],sr,profile['hz']),analytic_fundamental(candidate,sr,profile['hz']),blend)
                ramps=[ramp_metrics(mix,st,start,end,side,floor) for start,end in case['intervals'] for side in ('entry','exit')]
                arrays.update(sample_times=st[::8],**{k:v[::8] for k,v in mix.items()})
                npz=folder/(case['curve']+'.npz');np.savez_compressed(npz,**arrays);artifact(npz)
                wav=folder/(case['curve']+'.wav');sf.write(wav,candidate,sr,subtype='FLOAT')
                saved,rate=sf.read(wav,dtype='float32')
                if rate!=sr or not np.array_equal(saved,candidate):raise ValueError('Saved candidate changed')
                artifact(wav)
                original=next(c for c in boundary['cases'] if c['profile']==profile['name'] and c['curve']==case['curve'])
                record['cases'].append(dict(profile=profile['name'],curve=case['curve'],intervals=case['intervals'],
                    source_path=case['source_path'],baseline_candidate_path=case['candidate_path'],candidate_path=str(wav),arrays=str(npz),
                    renderer=renderer,preservation=preservation,audit=audit,baseline_audit=case['audit'],
                    transition_peaks=peaks,baseline_transition_peaks=original['baseline_transition_peaks'],phase_ramps=ramps))
                save()
        cases=record['cases'];label=lambda c:c['profile']+'/'+c['curve']
        summary=dict(tested=len(cases),baseline_passes=sum(c['baseline_audit']['passed'] for c in cases),candidate_passes=sum(c['audit']['passed'] for c in cases),
            candidate_fine_passes=sum(c['audit']['fine'] is not None and c['audit']['fine']['median_error_cents']<=5. and c['audit']['fine']['p90_error_cents']<=10. for c in cases),
            physical_passes=sum(all(c['preservation'].values()) for c in cases),identity_passes=len(record['identities']),
            lost_baseline_passes=[label(c) for c in cases if c['baseline_audit']['passed'] and not c['audit']['passed']],
            gained_passes=[label(c) for c in cases if not c['baseline_audit']['passed'] and c['audit']['passed']],transitions={})
        for w in ('2048','1024'):
            b=[c['baseline_transition_peaks'][w]['peak_abs_cents'] for c in cases];v=[c['transition_peaks'][w]['peak_abs_cents'] for c in cases]
            summary['transitions'][w]=dict(baseline_above30=sum(p>30 for p in b),candidate_above30=sum(p>30 for p in v),
                baseline_max_peak=max(b),candidate_max_peak=max(v),
                cases_peak_increased=[label(c) for c,x,y in zip(cases,b,v) if y>x],cases_peak_decreased=[label(c) for c,x,y in zip(cases,b,v) if y<x])
        ramps=[r for c in cases for r in c['phase_ramps']];supported=[r for r in ramps if not r['withheld']]
        summary['phase_diagnostic']=dict(ramps=len(ramps),supported_ramps=len(supported),
            ramps_measured_below_half=sum(r['minimum_measured_ratio']<.5 for r in supported),
            ramps_measured_below_point8=sum(r['minimum_measured_ratio']<.8 for r in supported),
            minimum_measured_ratio=min(r['minimum_measured_ratio'] for r in supported),baseline=phase['summary']['baseline'])
        record['summary']=summary;record['advancement']=advancement(summary)
        if any(checksum(path)!=digest for path,digest in tracked.items()) or checksum(parselmouth.__file__)!=record['backend_sha256'] or checksum(root/'declaration.json')!=record['declaration_sha256']:
            raise ValueError('Frozen source/declaration changed')
        record['run_status']='complete';save();print(json.dumps(dict(summary=summary,advancement=record['advancement']),indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
