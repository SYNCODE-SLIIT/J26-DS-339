"""Oracle-only constructed phase-anchored audio controls, never ordinary vocal repair."""
from bpdr_input import COMPONENT_ROOT
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.audio.analysis import extract
from bpdr_input.preparation.sources.data_builder import write_json
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.experiments.audio.framewise_diagnostics import relative_peak, transition_mask
from bpdr_input.experiments.audio.framewise_fixtures import constructed, signal, requested_curve
from bpdr_input.repair.interventions import AuditPolicy, envelope, independent_pitch
from bpdr_input.audio.pipeline import checksum

ANCHOR='runs/anchor_feasibility/fa470618ad02487d90cb4436c2864e47/evidence.json'
BASELINE='runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/evidence.json'
VERSION='known-source-anchor-control-0.1'


def strict_eligible(case):
    return bool(case['supports']) and all(s['sign_preserving']['phase_closed_solution'] is True for s in case['supports'])


def make_known_control(profile,audio,supports,sr=22050):
    """Requires construction F0/phase; refuses other waveforms and relaxed solutions.

    No source estimation, native vocoder, user-audio repair or accepted package is
    implemented here. Source phase plus the saved added cycles drives exactly the
    same harmonic family inside the supports; all outside samples are copied.
    """
    if sr!=22050 or audio.ndim!=1 or audio.dtype!=np.float32:raise ValueError('Known-control source format mismatch')
    st,f0,amp=constructed(profile,sr)
    if not np.array_equal(signal(f0,amp,sr),audio):raise ValueError('Source is not the declared constructed waveform')
    diagnostic=dict(scope='known_source_oracle_control_only',identity=not bool(supports),
        known_source_f0_and_phase_used=True,ordinary_vocal_renderer=False,production_eligible=False,accepted_audio=None)
    output=audio.copy()
    if not supports:return output,diagnostic
    phase=2*np.pi*np.cumsum(f0)/sr;previous=0.;checks=[]
    for start,end,a in supports:
        if not np.isfinite(start+end) or start<previous or not 0<=start<end<=len(audio)/sr or end-start<.86-1e-12:
            raise ValueError('Invalid known-control supports')
        previous=end;mask=(st>=start)&(st<end);n=int(mask.sum())
        keys=('times_seconds','source_hz','requested','sign_preserving_cents','sign_preserving_added_cycles','entry_mask','exit_mask')
        if any(k not in a or np.asarray(a[k]).shape!=(n,) for k in keys):raise ValueError('Missing strict solution; no relaxed fallback')
        if not np.array_equal(a['times_seconds'],st[mask]) or not np.array_equal(a['source_hz'],f0[mask]):raise ValueError('Wrong source/time mapping')
        requested=a['requested'];cents=a['sign_preserving_cents'];cycles=a['sign_preserving_added_cycles']
        if any(not np.isfinite(v).all() for v in (requested,cents,cycles)) or abs(requested).max()>30.:
            raise ValueError('Invalid known-control correction arrays')
        expected_entry=np.arange(n)<1764;expected_exit=np.arange(n)>=n-1764
        if not np.array_equal(a['entry_mask'],expected_entry) or not np.array_equal(a['exit_mask'],expected_exit):raise ValueError('Changed 80-ms ramps')
        interior=~(expected_entry|expected_exit)
        if (np.any(cents<np.minimum(0,requested)-1e-8) or np.any(cents>np.maximum(0,requested)+1e-8)
                or not np.allclose(cents[interior],requested[interior],rtol=0,atol=1e-8)
                or cents[0]!=0 or cents[-1]!=0):raise ValueError('Strict correction bound/interior/endpoint violation')
        added=f0[mask]*np.expm1(np.log(2)*cents/1200.)
        if not np.allclose(np.cumsum(added)/sr,cycles,rtol=0,atol=1e-10) or abs(cycles[-1]-round(float(cycles[-1])))>1e-10:
            raise ValueError('Frequency/phase mismatch or unclosed phase')
        if np.any(cents):
            local_phase=phase[mask]+2*np.pi*cycles
            local=(amp[mask]*(np.sin(local_phase)+.4*np.sin(2*local_phase)+.2*np.sin(3*local_phase))).astype(np.float32)
            if local[0]!=audio[mask][0] or local[-1]!=audio[mask][-1]:raise ValueError('Endpoint samples do not match; no forced replacement')
            output[mask]=local
        checks.append(dict(start_seconds=start,end_seconds=end,phase_residual_cycles=float(cycles[-1]-round(float(cycles[-1]))),
            endpoints_exact=bool(output[mask][0]==audio[mask][0] and output[mask][-1]==audio[mask][-1])))
    if not np.isfinite(output).all() or abs(output).max()>=1:raise ValueError('Nonfinite/clipped oracle control')
    diagnostic.update(supports=checks,sample_count_unchanged=output.shape==audio.shape)
    return output,diagnostic


def amplitude_metrics(source_analytic,baseline_analytic,control_analytic,st,intervals,source_voiced):
    """Source-relative ratios; not comparable to older full-branch denominators."""
    source_amp=abs(source_analytic);floor=.1*float(np.median(source_amp[source_voiced]))
    # The same source-supported mask is used for both outputs; output dips do not
    # remove samples from evaluation.
    ratios={name:np.divide(abs(z),source_amp,out=np.full(len(st),np.nan),where=source_amp>0)
            for name,z in [('baseline',baseline_analytic),('control',control_analytic)]}
    rows=[]
    for start,end in intervals:
        weight=envelope(st,(start,end),.08)
        for side,lo,hi in (('entry',start,start+.08),('exit',end-.08,end)):
            valid=(st>=lo)&(st<hi)&(weight>.1)&(weight<.9)&(source_amp>floor)
            if not valid.any():raise ValueError('No supported source-amplitude diagnostic samples')
            rows.append(dict(side=side,start_seconds=lo,end_seconds=hi,samples=int(valid.sum()),
                **{name:dict(minimum_ratio=float(np.min(r[valid])),below_half=bool(np.any(r[valid]<.5)),
                            below_point8=bool(np.any(r[valid]<.8))) for name,r in ratios.items()}))
    return rows,ratios


def run():
    import librosa
    import torch
    from bpdr_input.experiments.audio.phase_feasibility import analytic_fundamental
    component=COMPONENT_ROOT
    anchor_path=component/ANCHOR;anchor=json.loads(anchor_path.read_text(encoding='utf-8'))
    baseline_path=component/BASELINE;baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    if any(v['run_status']!='complete' or v['adopted'] for v in (anchor,baseline)):raise ValueError('Wrong prior evidence')
    if baseline['base_audit_limits']!=json.loads(json.dumps(asdict(AuditPolicy()))):raise ValueError('Audit policy changed')
    selected=[dict(profile=c['profile'],curve=c['curve']) for c in anchor['cases'] if strict_eligible(c)]
    withheld=[dict(profile=c['profile'],curve=c['curve']) for c in anchor['cases'] if not strict_eligible(c)]
    tracked=dict(anchor['tracked_inputs']);tracked[str(anchor_path)]=checksum(anchor_path)
    for a in anchor['artifacts']:tracked[a['path']]=a['sha256']
    own=Path(__file__);tracked[str(own)]=checksum(own)
    for path,digest in tracked.items():
        if checksum(path)!=digest:raise ValueError('Changed input: '+path)
    root=component/'runs/known_source_control'/uuid.uuid4().hex;root.mkdir(parents=True)
    snapshots=root/'source_code_at_run';snapshots.mkdir()
    for path in tracked:
        if Path(path).suffix=='.py':(snapshots/Path(path).name).write_bytes(Path(path).read_bytes())
    declaration=dict(date='2026-10-15',gate='Gate F — bounded known-source analytical audio control',selected=selected,withheld=withheld,
        selection='Entire case eligible only when every support has the frozen strict fixed-basis solution; no outcome-based change',
        synthesis='Same known harmonic family and source amplitude; original source phase plus saved strict added cycles only in supports',
        leakage_scope='Construction F0/phase are oracle inputs to synthesis only; PESTO/YIN auditors receive WAV samples, no oracle target F0',
        audit='Unchanged framewise-render-audit-0.1; original framewise requested curve, both YIN windows 2048/1024',
        preservation='Exact zero identities, source count/rate/time mapping, outside/rest/endpoint samples; FLOAT WAV round trip exact',
        amplitude='Fixed known-center filter, source-relative amplitude ratios for both baseline/control on the same source-supported masks',
        amplitude_comparison='New source-relative reference; do not compare directly to older phase-aligned branch denominators',
        equation_checks='Frozen feasibility numerical checks 1e-10 cycles and 1e-8 cents; no audit epsilon or musical threshold change',
        tuning='Use saved curves without changes, no relaxed solution or rerender after outcomes, no exclusions among selected cases',
        constraints='Oracle control, not a practical vocal renderer, adopted permission, training example or accepted delivery',
        musical_review='Gate D/J4 deferred by owner; engineering continues',adopted=False,training_ready_examples=0)
    write_json(root/'declaration.json',declaration)
    record=dict(run_status='running',version=VERSION,data_status='exploratory_not_research',declaration=declaration,
        declaration_sha256=checksum(root/'declaration.json'),tracked_inputs=tracked,cpu_threads=torch.get_num_threads(),
        cases=[],identities=[],artifacts=[],adopted=False,ordinary_vocal_renderer=False,accepted_prepared_packages=0,training_ready_examples=0)
    def save():write_json(root/'evidence.json',record)
    def artifact(path):record['artifacts'].append(dict(path=str(path),sha256=checksum(path)))
    save();print('KNOWN_SOURCE_CONTROL_RUN:',root,flush=True)
    try:
        for profile in baseline['profiles']:
            print('PROFILE:',profile['name'],flush=True)
            cases=[c for c in anchor['cases'] if c['profile']==profile['name']]
            audio,sr=sf.read(cases[0]['source_path'],dtype='float32');st=np.arange(len(audio))/sr
            identity,diag=make_known_control(profile,audio,[],sr)
            if not np.array_equal(identity,audio):raise ValueError('Zero identity failed')
            record['identities'].append(dict(profile=profile['name'],exact=True))
            before,analyzer=extract(audio,sr,component/'checkpoints');times=before['times_seconds'];y0=independent_pitch(audio,sr)
            original=next(c for c in baseline['cases'] if c['profile']==profile['name'])
            if analyzer!=original['source_analyzer']:raise ValueError('Analyzer/checkpoint changed')
            reliable=before['voiced_mask']&np.isfinite(y0)&(y0>0)
            reliable[reliable]&=abs(1200*np.log2(y0[reliable]/before['f0_hz'][reliable]))<=AuditPolicy().tracker_agreement_cents
            if profile['kind']=='rest':reliable&=(times<2.5)|(times>=4.5)
            refs={2048:y0,1024:librosa.yin(audio,sr=sr,fmin=50.,fmax=1500.,frame_length=1024,hop_length=441)}
            source_analytic=analytic_fundamental(audio,sr,profile['hz'])
            for case in cases:
                if not strict_eligible(case):
                    record['cases'].append(dict(profile=profile['name'],curve=case['curve'],outcome='WITHHELD_NO_STRICT_SOLUTION',
                        rendered=False,candidate_path=None,reasons=['No strict fixed-basis solution for every support'],
                        mathematical_supports=[dict(start=s['start_seconds'],end=s['end_seconds'],strict=s['sign_preserving']) for s in case['supports']]))
                    save();continue
                prepared=[]
                for s in case['supports']:
                    with np.load(s['arrays']) as a:arrays={k:a[k] for k in a.files}
                    prepared.append((s['start_seconds'],s['end_seconds'],arrays))
                candidate,diagnostic=make_known_control(profile,audio,prepared,sr)
                intervals=[(s['start_seconds'],s['end_seconds']) for s in case['supports']]
                inside=np.zeros(len(audio),dtype=bool)
                for start,end in intervals:inside|=(st>=start)&(st<end)
                preservation=dict(sample_count=candidate.shape==audio.shape,outside_exact=bool(np.array_equal(candidate[~inside],audio[~inside])),
                    rests_exact=bool(np.array_equal(candidate[(st>=2.5)&(st<4.5)],audio[(st>=2.5)&(st<4.5)])) if profile['kind']=='rest' else True,
                    endpoints_exact=all(s['endpoints_exact'] for s in diagnostic['supports']),finite_unclipped=bool(np.isfinite(candidate).all() and abs(candidate).max()<1))
                if not all(preservation.values()):raise ValueError('Physical preservation failed')
                wav=root/(profile['name']+'_'+case['curve']+'.wav');sf.write(wav,candidate,sr,subtype='FLOAT')
                saved,rate=sf.read(wav,dtype='float32')
                if rate!=sr or not np.array_equal(saved,candidate):raise ValueError('Saved FLOAT WAV changed')
                artifact(wav)
                after,_=extract(saved,sr,component/'checkpoints');y1=independent_pitch(saved,sr)
                original_case=next(c for c in baseline['cases'] if c['profile']==profile['name'] and c['curve']==case['curve'])
                requested=requested_curve(times,intervals,case['curve'])
                audit=audit_curve(before,after,y0,y1,reliable,intervals,requested)
                baseline_audio,rate=sf.read(original_case['candidate_path'],dtype='float32')
                if rate!=sr:raise ValueError('Baseline rate changed')
                mask=transition_mask(times,intervals);peaks={};out=dict(times=times,requested=requested,transition_mask=mask)
                for w in refs:
                    yc=y1 if w==2048 else librosa.yin(saved,sr=sr,fmin=50.,fmax=1500.,frame_length=w,hop_length=441)
                    yb=librosa.yin(baseline_audio,sr=sr,fmin=50.,fmax=1500.,frame_length=w,hop_length=441)
                    cp,out[f'control_transition_delta_{w}']=relative_peak(refs[w],yc,mask)
                    bp,out[f'baseline_transition_delta_{w}']=relative_peak(refs[w],yb,mask)
                    peaks[str(w)]=dict(control=cp,baseline=bp)
                source_voiced=(st<2.5)|(st>=4.5) if profile['kind']=='rest' else np.ones(len(st),dtype=bool)
                amp_rows,ratios=amplitude_metrics(source_analytic,analytic_fundamental(baseline_audio,sr,profile['hz']),analytic_fundamental(saved,sr,profile['hz']),st,intervals,source_voiced)
                out.update(sample_times=st[::8],**{name+'_source_amplitude_ratio':r[::8] for name,r in ratios.items()})
                npz=root/(profile['name']+'_'+case['curve']+'.npz');np.savez_compressed(npz,**out);artifact(npz)
                record['cases'].append(dict(profile=profile['name'],curve=case['curve'],outcome='ORACLE_CONTROL_MEASURED',rendered=True,
                    source_path=case['source_path'],baseline_candidate_path=original_case['candidate_path'],candidate_path=str(wav),arrays=str(npz),
                    intervals=intervals,construction=diagnostic,preservation=preservation,audit=audit,baseline_audit=original_case['audit'],
                    transition_peaks=peaks,source_relative_amplitude=amp_rows))
                save()
        measured=[c for c in record['cases'] if c['rendered']];ramps=[r for c in measured for r in c['source_relative_amplitude']]
        summary=dict(total_cases=len(record['cases']),measured=len(measured),withheld=sum(not c['rendered'] for c in record['cases']),
            full_audit_passes=sum(c['audit']['passed'] for c in measured),matched_baseline_passes=sum(c['baseline_audit']['passed'] for c in measured),
            fine_error_passes=sum(c['audit']['fine'] is not None and c['audit']['fine']['median_error_cents']<=5 and c['audit']['fine']['p90_error_cents']<=10 for c in measured),
            physical_passes=sum(all(c['preservation'].values()) for c in measured),identity_passes=len(record['identities']),
            transitions={w:{name:dict(above30=sum(c['transition_peaks'][w][name]['peak_abs_cents']>30 for c in measured),
                max_peak=max(c['transition_peaks'][w][name]['peak_abs_cents'] for c in measured)) for name in ('baseline','control')} for w in ('2048','1024')},
            amplitude=dict(reference='source fundamental, not the older weighted-branch denominator',ramps=len(ramps),
                **{name:dict(ramps_below_half=sum(r[name]['below_half'] for r in ramps),ramps_below_point8=sum(r[name]['below_point8'] for r in ramps),
                    minimum_ratio=min(r[name]['minimum_ratio'] for r in ramps)) for name in ('baseline','control')}))
        record['summary']=summary
        if any(checksum(p)!=h for p,h in tracked.items()) or checksum(root/'declaration.json')!=record['declaration_sha256']:raise ValueError('Frozen source/declaration changed')
        record['run_status']='complete';save();print(json.dumps(summary,indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
