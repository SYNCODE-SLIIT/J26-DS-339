"""Bounded 20-request audio-only correction experiment, without policy adoption."""
from bpdr_input import COMPONENT_ROOT, module_source
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.audio.analysis import extract
from bpdr_input.preparation.sources.data_builder import write_json
from bpdr_input.experiments.audio.estimated_harmonic_renderer import render_estimated_candidate
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.experiments.audio.framewise_diagnostics import direction_details, relative_peak, transition_mask
from bpdr_input.experiments.audio.framewise_fixtures import CURVES, requested_curve
from bpdr_input.repair.interventions import AuditPolicy, independent_pitch
from bpdr_input.experiments.audio.known_source_control import amplitude_metrics
from bpdr_input.experiments.audio.phase_feasibility import analytic_fundamental
from bpdr_input.audio.pipeline import checksum
from bpdr_input.repair.repair_controller import waveform_hash

SEPARATION='runs/harmonic_separation/5e708a6b633d49dd9de7a0958783c617/evidence.json'
ESTIMATION='runs/source_estimation/8c34deae8796487e9e261cd41bff9171/evidence.json'
BASELINE='runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/evidence.json'
VERSION='estimated-harmonic-correction-experiment-0.1'


def run():
    import librosa
    import torch
    component=COMPONENT_ROOT;sep_path=component/SEPARATION;est_path=component/ESTIMATION
    prior=json.loads(sep_path.read_text(encoding='utf-8'));est=json.loads(est_path.read_text(encoding='utf-8'))
    baseline=json.loads((component/BASELINE).read_text(encoding='utf-8'))
    if prior['run_status']!='complete' or prior['adopted'] or est['run_status']!='complete':raise ValueError('Wrong frozen inputs')
    if baseline['base_audit_limits']!=json.loads(json.dumps(asdict(AuditPolicy()))):raise ValueError('Audit policy changed')
    selected=[c for c in prior['sources'] if c['status']=='decomposed']
    source_withheld=[dict(profile=c['profile'],status=c['status'],reasons=c['reasons']) for c in prior['sources'] if c['status']!='decomposed']
    tracked=dict(prior['tracked_inputs']);tracked[str(sep_path)]=checksum(sep_path);tracked[str(est_path)]=checksum(est_path)
    for item in prior['artifacts']:tracked[item['path']]=item['sha256']
    for p in (Path(__file__),module_source('estimated_harmonic_renderer.py')):tracked[str(p)]=checksum(p)
    for p,h in tracked.items():
        if checksum(p)!=h:raise ValueError('Frozen input changed: '+p)
    root=component/'runs/estimated_harmonic_correction'/uuid.uuid4().hex;root.mkdir(parents=True)
    snapshots=root/'source_code_at_run';snapshots.mkdir()
    for p in tracked:
        if Path(p).suffix=='.py':(snapshots/Path(p).name).write_bytes(Path(p).read_bytes())
    declaration=dict(date='2026-10-15',gate='Gate F — bounded audio-only estimated-harmonic correction',
        requests=[dict(profile=c['profile'],curve=curve) for c in selected for curve in CURVES],withheld_source_profiles=source_withheld,
        selection='All five decomposed profiles x four frozen curves =20 requests; prior three source refusals remain withheld, not twelve additional measured requests',
        runtime_inputs='Saved source samples, frozen audio-only estimated phase/frequency/support, source-fitted six coefficients/residual and supplied curve; no construction values',
        mathematical_kernel='Unchanged solve_support on estimated sample frequency and original interpolated request; strict solution only, entire request withheld if any support fails',
        bounds='80-ms ramps, zero endpoint correction, fixed interior request, ramp cents between zero and original supplied cents, no relaxed fallback',
        rendering='Source-fitted harmonics evaluated at estimated phase plus integrated strict frequency offset; original float64 residual added unchanged, outside support copied',
        closure='Final added cycles integer within original 1e-10 equation check; actual endpoint samples must match exactly without replacement',
        audit='Original framewise-render-audit-0.1 against original supplied frame curve; FLOAT WAV saved/reloaded exactly before analysis, unchanged PESTO provenance and policy',
        diagnostics='Both fixed YIN windows 2048/1024; raw source-relative change magnitudes, not target errors/audible thresholds; save opposing frames',
        amplitude='Fourth-order band at 0.7–1.3 times saved estimated center, source-relative denominator, same source-supported masks for matched baseline/candidate; no known center',
        comparison='Baseline only on identical rendered profile/curve subset; preserve gained/lost passes and every individual transition regression; no extrapolation to20 or32',
        scope='Simple constructed signals and supplied interventions, no validated prediction permission, uncertainty, natural-vocal transfer, learned benefit or musical review',
        tuning='One fixed run; no new source estimates/fits, threshold changes, epsilon, outcome-based rerender or parameter tuning',
        advancement='Descriptive bounded results only; no adoption or gate closure regardless of aggregate gains; unsupported requests and source refusals remain',
        adopted=False,training_ready_examples=0,accepted_prepared_packages=0)
    write_json(root/'declaration.json',declaration)
    record=dict(run_status='running',version=VERSION,data_status='exploratory_not_research',declaration=declaration,
        declaration_sha256=checksum(root/'declaration.json'),tracked_inputs=tracked,artifacts=[],cases=[],identities=[],
        withheld_source_profiles=source_withheld,cpu_threads=torch.get_num_threads(),adopted=False,training_ready_examples=0,accepted_prepared_packages=0)
    def save():write_json(root/'evidence.json',record)
    def artifact(p):record['artifacts'].append(dict(path=str(p),sha256=checksum(p)))
    save();print('ESTIMATED_HARMONIC_RUN:',root,flush=True)
    try:
        for number,source in enumerate(selected):
            print('PROFILE:',source['profile'],flush=True)
            original=next(c for c in est['sources'] if c['profile']==source['profile'])
            audio,sr=sf.read(source['source_path'],dtype='float32')
            with np.load(original['prediction_arrays']) as a:prediction={k:a[k] for k in a.files}
            with np.load(source['arrays']) as a:decomp={k:a[k] for k in a.files}
            p=dict(status=original['prediction']['status'],arrays=prediction)
            d=dict(status=source['status'],coefficients=source['decomposition']['coefficients'],arrays=decomp)
            residual_hash=waveform_hash(decomp['residual'])
            before,analyzer=extract(audio,sr,component/'checkpoints');times=before['times_seconds'];y0=independent_pitch(audio,sr)
            if analyzer!=original['prediction']['analyzer']:raise ValueError('Frozen analyzer/checkpoint changed')
            identity,diag,_=render_estimated_candidate(audio,sr,p,d,times,np.zeros(len(times)),[])
            if not np.array_equal(identity,audio) or not diag['identity']:raise ValueError('Zero identity failed')
            record['identities'].append(dict(profile=source['profile'],exact=True))
            reliable=before['voiced_mask']&np.isfinite(y0)&(y0>0)
            reliable[reliable]&=abs(1200*np.log2(y0[reliable]/before['f0_hz'][reliable]))<=AuditPolicy().tracker_agreement_cents
            refs={2048:y0,1024:librosa.yin(audio,sr=sr,fmin=50.,fmax=1500.,frame_length=1024,hop_length=441)}
            estimated_center=original['prediction']['estimated_center_hz'];source_analytic=analytic_fundamental(audio,sr,estimated_center)
            for curve in CURVES:
                base=next(c for c in baseline['cases'] if c['profile']==source['profile'] and c['curve']==curve)
                intervals=base['intervals'];request=requested_curve(times,intervals,curve)
                candidate,renderer,prepared=render_estimated_candidate(audio,sr,p,d,times,request,intervals)
                plan=root/(source['profile']+'_'+curve+'_plan.npz');np.savez_compressed(plan,**prepared);artifact(plan)
                case=dict(profile=source['profile'],curve=curve,intervals=intervals,source_path=source['source_path'],
                    baseline_candidate_path=base['candidate_path'],candidate_path=None,rendered=candidate is not None,
                    renderer=renderer,plan_arrays=str(plan))
                if candidate is None:record['cases'].append(case);save();continue
                wav=root/(source['profile']+'_'+curve+'.wav');sf.write(wav,candidate,sr,subtype='FLOAT')
                saved,rate=sf.read(wav,dtype='float32')
                if rate!=sr or not np.array_equal(saved,candidate):raise ValueError('Saved FLOAT WAV changed')
                artifact(wav)
                inside=prepared['inside_support'];preservation=dict(sample_count=saved.shape==audio.shape,
                    sample_rate=rate==sr,time_mapping=bool(np.array_equal(np.arange(len(saved)//441+1)*441/sr,times)),outside_exact=bool(np.array_equal(saved[~inside],audio[~inside])),
                    endpoints_exact=renderer['endpoints_exact'],residual_input_unchanged=waveform_hash(decomp['residual'])==residual_hash,
                    finite_unclipped=bool(np.isfinite(saved).all() and abs(saved).max()<1))
                if not all(preservation.values()):raise ValueError('Physical/residual preservation failed')
                after,after_analyzer=extract(saved,sr,component/'checkpoints');y1=independent_pitch(saved,sr)
                if after_analyzer!=analyzer:raise ValueError('Candidate analyzer changed')
                audit=json.loads(json.dumps(audit_curve(before,after,y0,y1,reliable,intervals,request)))
                opposing,detail=direction_details(before,after,y0,y1,reliable,intervals,request)
                mask=transition_mask(times,intervals);out=dict(transition_mask=mask,**detail);peaks={}
                old_audio,old_rate=sf.read(base['candidate_path'],dtype='float32')
                if old_rate!=sr:raise ValueError('Baseline rate changed')
                for w in refs:
                    yc=y1 if w==2048 else librosa.yin(saved,sr=sr,fmin=50.,fmax=1500.,frame_length=w,hop_length=441)
                    yb=librosa.yin(old_audio,sr=sr,fmin=50.,fmax=1500.,frame_length=w,hop_length=441)
                    cp,out[f'candidate_delta_{w}']=relative_peak(refs[w],yc,mask)
                    bp,out[f'baseline_delta_{w}']=relative_peak(refs[w],yb,mask);peaks[str(w)]=dict(candidate=cp,baseline=bp)
                amps,ratios=amplitude_metrics(source_analytic,analytic_fundamental(old_audio,sr,estimated_center),analytic_fundamental(saved,sr,estimated_center),
                    prediction['sample_times'],intervals,prediction['sample_reliable'])
                out.update(sample_times=prediction['sample_times'][::8],**{k+'_amplitude_ratio':v[::8] for k,v in ratios.items()})
                array=root/(source['profile']+'_'+curve+'_audit.npz');np.savez_compressed(array,**out);artifact(array)
                case.update(candidate_path=str(wav),audit_arrays=str(array),audit=audit,baseline_audit=base['audit'],
                    preservation=preservation,opposing_frames=opposing,transition_peaks=peaks,source_relative_amplitude=amps,
                    diagnostic_estimated_center_hz=estimated_center,residual_sha256=residual_hash)
                record['cases'].append(case);save()
        measured=[c for c in record['cases'] if c['rendered']];label=lambda c:c['profile']+'/'+c['curve']
        transitions={}
        for w in ('2048','1024'):
            transitions[w]={name:dict(above30=sum(c['transition_peaks'][w][name]['peak_abs_cents']>30 for c in measured),
                max_peak=max((c['transition_peaks'][w][name]['peak_abs_cents'] for c in measured),default=None)) for name in ('baseline','candidate')}
            transitions[w]['cases_peak_increased']=[label(c) for c in measured if c['transition_peaks'][w]['candidate']['peak_abs_cents']>c['transition_peaks'][w]['baseline']['peak_abs_cents']]
        ramps=[r for c in measured for r in c['source_relative_amplitude']]
        record['summary']=dict(requested_cases=len(record['cases']),rendered=len(measured),withheld_requests=sum(not c['rendered'] for c in record['cases']),
            prior_source_refusals=len(source_withheld),candidate_full_passes=sum(c['audit']['passed'] for c in measured),
            matched_baseline_full_passes=sum(c['baseline_audit']['passed'] for c in measured),
            candidate_fine_passes=sum(c['audit']['fine'] is not None and c['audit']['fine']['median_error_cents']<=5 and c['audit']['fine']['p90_error_cents']<=10 for c in measured),
            physical_passes=sum(all(c['preservation'].values()) for c in measured),identity_passes=len(record['identities']),
            lost_baseline_passes=[label(c) for c in measured if c['baseline_audit']['passed'] and not c['audit']['passed']],
            gained_baseline_passes=[label(c) for c in measured if not c['baseline_audit']['passed'] and c['audit']['passed']],transitions=transitions,
            amplitude=dict(ramps=len(ramps),**{name:dict(below_half=sum(r[name]['below_half'] for r in ramps),
                below_point8=sum(r[name]['below_point8'] for r in ramps),minimum_ratio=min((r[name]['minimum_ratio'] for r in ramps),default=None)) for name in ('baseline','control')}))
        if any(checksum(p)!=h for p,h in tracked.items()) or checksum(root/'declaration.json')!=record['declaration_sha256']:raise ValueError('Frozen input/declaration changed')
        record['run_status']='complete';save();print(json.dumps(record['summary'],indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
