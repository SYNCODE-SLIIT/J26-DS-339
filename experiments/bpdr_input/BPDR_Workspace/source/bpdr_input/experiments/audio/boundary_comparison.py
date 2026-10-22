"""Predeclared all-case boundary-taper comparison; no tuning, fitting or adoption."""
from bpdr_input import COMPONENT_ROOT, module_source
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.audio.analysis import extract
from bpdr_input.experiments.audio.boundary_renderer import VERSION, render_boundary_candidate
from bpdr_input.preparation.sources.data_builder import write_json
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.experiments.audio.framewise_diagnostics import relative_peak, transition_mask
from bpdr_input.experiments.audio.framewise_fixtures import requested_curve
from bpdr_input.repair.framewise_renderer import RendererPolicy
from bpdr_input.repair.interventions import AuditPolicy, independent_pitch
from bpdr_input.audio.pipeline import checksum
from bpdr_input.reporting.notebook_evidence import component_path

BASELINE='runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/evidence.json'
DIAGNOSTIC='runs/framewise_diagnostics/4ece4923dc68495ba7efd2580733d89e/evidence.json'
DECLARATION=dict(gate='Gate F — usable delivered audio / controlled boundary handling',
    scope='Constructed engineering comparison, not vocal repair or musical approval',
    selection='All eight existing profiles x four existing curves, unchanged supports and sample sources',
    method='One candidate: 80-ms raised-cosine taper on the original 20-ms correction curve before native-tier interpolation; original local waveform envelope unchanged',
    hypothesis='A smoother native-tier correction at boundaries may reduce transition estimator peaks while retaining interior targets',
    audit='Unchanged framewise-render-audit-0.1 against ORIGINAL requested interior curve; no coarse epsilon',
    transition_windows_samples=[2048,1024],
    transition_definition='Existing [start-.10,start+.18) union (end-.18,end+.10] windows; estimator observations only',
    advancement_requirements=['All 32 physical/count/time/outside/rest checks and eight zero identities',
        'All 32 existing fine-error checks', 'No loss of any baseline full-audit pass',
        'Neither count of transition peaks >30 nor maximum peak worsens at either fixed YIN window',
        'At least one of those four transition summary observations improves'],
    advancement_meaning='Engineering filter for considering a bounded training-vocal/listening comparison; not audio acceptance or research eligibility',
    tuning='No parameter search, rerun with changed settings, case exclusions or favorable-window selection after outcomes',
    adopted=False,accepted_prepared_packages=0,training_ready_examples=0)


def advancement(summary):
    checks=dict(physical_preservation=summary['physical_passes']==32 and summary['identity_passes']==8,
        fine_checks=summary['candidate_fine_passes']==32,
        no_baseline_pass_lost=not summary['lost_baseline_passes'],
        transition_summary_nonregression=all(v['candidate_above30']<=v['baseline_above30'] and v['candidate_max_peak']<=v['baseline_max_peak'] for v in summary['transitions'].values()),
        transition_summary_improvement=any(v['candidate_above30']<v['baseline_above30'] or v['candidate_max_peak']<v['baseline_max_peak'] for v in summary['transitions'].values()))
    return dict(checks=checks,eligible_for_bounded_vocal_comparison=all(checks.values()),
                adopted=False,meaning=DECLARATION['advancement_meaning'])


def run():
    import librosa
    import parselmouth
    import torch
    component=COMPONENT_ROOT
    baseline_path=component/BASELINE;diag_path=component/DIAGNOSTIC
    baseline=json.loads(baseline_path.read_text(encoding='utf-8'));diag=json.loads(diag_path.read_text(encoding='utf-8'))
    if any(v['run_status']!='complete' or v['adopted'] for v in (baseline,diag)):raise ValueError('Invalid baseline evidence')
    if checksum(baseline_path)!=diag['source_evidence_sha256']:raise ValueError('Baseline changed')
    if baseline['renderer_policy']!=asdict(RendererPolicy()) or baseline['base_audit_limits']!=json.loads(json.dumps(asdict(AuditPolicy()))):raise ValueError('Policy changed')
    tracked=dict(diag['tracked_inputs']);tracked[str(diag_path)]=checksum(diag_path)
    for item in diag['artifacts']:tracked[str(component_path(component,item['path']))]=item['sha256']
    names=('boundary_renderer','boundary_comparison','framewise_diagnostics','pipeline','notebook_evidence')
    for name in names:tracked[str(module_source(name+'.py'))]=checksum(module_source(name+'.py'))
    for path,digest in tracked.items():
        if checksum(path)!=digest:raise ValueError('Changed input: '+path)
    if checksum(parselmouth.__file__)!=baseline['backend_sha256']:raise ValueError('Changed native backend')
    root=component/'runs/boundary_comparison'/uuid.uuid4().hex;root.mkdir(parents=True)
    snapshots=root/'source_code_at_run';snapshots.mkdir()
    for path,digest in tracked.items():
        if Path(path).suffix=='.py':(snapshots/Path(path).name).write_bytes(Path(path).read_bytes())
    declaration=root/'declaration.json';write_json(declaration,DECLARATION)
    record=dict(run_status='running',data_status='exploratory_not_research',candidate_version=VERSION,
        declaration_sha256=checksum(declaration),declaration=DECLARATION,tracked_inputs=tracked,
        backend_path=parselmouth.__file__,backend_sha256=baseline['backend_sha256'],
        parselmouth_version=parselmouth.__version__,praat_version=parselmouth.PRAAT_VERSION,
        cpu_threads=torch.get_num_threads(),cases=[],identities=[],artifacts=[],adopted=False,
        accepted_prepared_packages=0,training_ready_examples=0)
    def save():write_json(root/'evidence.json',record)
    def artifact(path):record['artifacts'].append(dict(path=str(path),sha256=checksum(path)))
    save();print('BOUNDARY_COMPARISON_RUN:',root,flush=True)
    try:
        for profile in baseline['profiles']:
            print('PROFILE:',profile['name'],flush=True)
            cases=[c for c in baseline['cases'] if c['profile']==profile['name']]
            audio,sr=sf.read(cases[0]['source_path'],dtype='float32')
            if sr!=22050:raise ValueError('Source rate changed')
            before,analyzer=extract(audio,sr,component/'checkpoints')
            if analyzer!=cases[0]['source_analyzer']:raise ValueError('Analyzer/checkpoint changed')
            times=before['times_seconds'];yin=independent_pitch(audio,sr)
            reliable=before['voiced_mask']&np.isfinite(yin)&(yin>0)
            reliable[reliable]&=abs(1200*np.log2(yin[reliable]/before['f0_hz'][reliable]))<=AuditPolicy().tracker_agreement_cents
            if profile['kind']=='rest':reliable&=(times<2.5)|(times>=4.5)
            identity,identity_diag,_=render_boundary_candidate(audio,sr,times,np.zeros(len(times)),[])
            if not np.array_equal(identity,audio) or 'praat_version' in identity_diag:raise ValueError('Zero bypass changed')
            record['identities'].append(dict(profile=profile['name'],exact=True,backend_bypassed=True))
            refs={2048:yin,1024:librosa.yin(audio,sr=sr,fmin=50.,fmax=1500.,frame_length=1024,hop_length=441)}
            folder=root/profile['name'];folder.mkdir()
            for case in cases:
                curve=requested_curve(times,case['intervals'],case['curve'])
                candidate,render_diag,tapered=render_boundary_candidate(audio,sr,times,curve,case['intervals'])
                st=np.arange(len(audio))/sr;inside=np.zeros(len(audio),dtype=bool);plateau=np.zeros(len(times),dtype=bool)
                for start,end in case['intervals']:
                    inside|=(st>start)&(st<end);plateau|=(times>=start+.18)&(times<=end-.18)
                preservation=dict(sample_count=candidate.shape==audio.shape,
                    time_origin=render_diag['source_time_origin_seconds']==render_diag['output_time_origin_seconds'],
                    duration_tier_unchanged=not render_diag['duration_tier_modified'],
                    outside_exact=bool(np.array_equal(candidate[~inside],audio[~inside])),
                    rests_exact=bool(np.array_equal(candidate[(st>=2.5)&(st<4.5)],audio[(st>=2.5)&(st<4.5)])) if profile['kind']=='rest' else True,
                    finite_unclipped=bool(np.isfinite(candidate).all() and abs(candidate).max()<1.),
                    interior_request_unchanged=bool(np.array_equal(tapered[plateau],curve[plateau])))
                if not all(preservation.values()):raise ValueError('Physical or target preservation failed')
                after,_=extract(candidate,sr,component/'checkpoints');y1=independent_pitch(candidate,sr)
                audited=audit_curve(before,after,yin,y1,reliable,case['intervals'],curve)
                mask=transition_mask(times,case['intervals']);peaks={}
                arrays=dict(times=times,original_requested=curve,tapered_requested=tapered,transition_mask=mask)
                for w in refs:
                    y=y1 if w==2048 else librosa.yin(candidate,sr=sr,fmin=50.,fmax=1500.,frame_length=w,hop_length=441)
                    peaks[str(w)],arrays[f'transition_delta_{w}']=relative_peak(refs[w],y,mask)
                wav=folder/(case['curve']+'.wav');sf.write(wav,candidate,sr,subtype='FLOAT')
                loaded,rate=sf.read(wav,dtype='float32')
                if rate!=sr or not np.array_equal(loaded,candidate):raise ValueError('Saved waveform changed')
                artifact(wav);npz=folder/(case['curve']+'.npz');np.savez_compressed(npz,**arrays);artifact(npz)
                original_diag=next(c for c in diag['cases'] if c['profile']==profile['name'] and c['curve']==case['curve'])
                record['cases'].append(dict(profile=profile['name'],curve=case['curve'],intervals=case['intervals'],
                    source_path=case['source_path'],baseline_candidate_path=case['candidate_path'],candidate_path=str(wav),
                    arrays=str(npz),renderer=render_diag,preservation=preservation,audit=audited,baseline_audit=case['audit'],
                    transition_peaks=peaks,baseline_transition_peaks={w:v['local'] for w,v in original_diag['transition_peaks'].items()}))
                save()
        selected=record['cases'];label=lambda c:c['profile']+'/'+c['curve']
        summary=dict(tested=len(selected),baseline_passes=sum(c['baseline_audit']['passed'] for c in selected),
            candidate_passes=sum(c['audit']['passed'] for c in selected),
            candidate_fine_passes=sum(c['audit']['fine'] is not None and c['audit']['fine']['median_error_cents']<=5. and c['audit']['fine']['p90_error_cents']<=10. for c in selected),
            lost_baseline_passes=[label(c) for c in selected if c['baseline_audit']['passed'] and not c['audit']['passed']],
            gained_passes=[label(c) for c in selected if not c['baseline_audit']['passed'] and c['audit']['passed']],
            physical_passes=sum(all(c['preservation'].values()) for c in selected),identity_passes=len(record['identities']),transitions={})
        for w in ('2048','1024'):
            b=[c['baseline_transition_peaks'][w]['peak_abs_cents'] for c in selected];v=[c['transition_peaks'][w]['peak_abs_cents'] for c in selected]
            summary['transitions'][w]=dict(baseline_above30=sum(p>30 for p in b),candidate_above30=sum(p>30 for p in v),
                baseline_max_peak=max(b),candidate_max_peak=max(v),
                cases_peak_increased=[label(c) for c,x,y in zip(selected,b,v) if y>x],
                cases_peak_decreased=[label(c) for c,x,y in zip(selected,b,v) if y<x])
        record['summary']=summary;record['advancement']=advancement(summary)
        if any(checksum(path)!=digest for path,digest in tracked.items()) or checksum(parselmouth.__file__)!=record['backend_sha256'] or checksum(declaration)!=record['declaration_sha256']:
            raise ValueError('Frozen source/declaration changed during comparison')
        record['run_status']='complete';save();print(json.dumps(dict(summary=summary,advancement=record['advancement']),indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
