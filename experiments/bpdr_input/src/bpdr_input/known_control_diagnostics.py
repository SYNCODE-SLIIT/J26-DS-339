"""Read-only replay of fixed oracle WAVs; no renderer or acceptance-policy changes."""
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from .analysis import extract
from .data_builder import write_json
from .framewise_audit import audit_curve
from .framewise_diagnostics import direction_details, relative_peak, transition_mask
from .framewise_fixtures import constructed, signal, requested_curve
from .interventions import AuditPolicy, cents_difference, independent_pitch
from .pipeline import checksum

SOURCE = 'runs/known_source_control/1d7a2f6206d94eb5993cfe2cbd0b56f2/evidence.json'
ANCHOR = 'runs/anchor_feasibility/fa470618ad02487d90cb4436c2864e47/evidence.json'
BASELINE = 'runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/evidence.json'
VERSION = 'known-control-diagnostics-0.1'
DECLARATION = dict(
    date='2026-10-15', gate='Gate F — frozen known-source audio diagnosis',
    selection='All eight previously measured cases, both prior baseline/control methods, both fixed YIN windows; retain 24 withheld rows',
    replay='Reproduce all 16 complete guarded audit dictionaries and all 32 saved transition peaks; stop on disagreement',
    direction='Retain exact opposing PESTO values, native-to-aligned mapping, float32 and float64 ratio differences; no epsilon',
    transition='Existing abs(YIN candidate/source change) is a change magnitude, not a target-error or artifact score',
    new_diagnostic='For oracle control only, compare YIN change to the saved strict sample-law cents at the nominal frame center; baseline has no known actual output law',
    residual_scope='Center-law residual uses construction truth for diagnosis only; finite-window tracker outputs need not equal instantaneous center frequency',
    identity='Zero observed change and zero law give zero residual; an exactly intended shift has nonzero raw change and zero residual',
    peak_context='Report both the original raw-change peak and the maximum center-law residual; do not swap locations or redefine old peaks',
    tuning='No new WAV, rerender, selection, curve adjustment, window substitution, acceptance threshold or tracker retraining',
    musical_review='Gate D/J4 deferred by owner; numerical diagnosis does not approve music',
    adopted=False, training_ready_examples=0, accepted_prepared_packages=0)


def strict_sample_law(sample_times, supports):
    """Load the exact frozen control law on its sample grid; no relaxed fallback."""
    t=np.asarray(sample_times)
    if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Invalid source sample grid')
    result=np.zeros(len(t));used=np.zeros(len(t),dtype=bool)
    for start,end,a in supports:
        mask=(t>=start)&(t<end)
        if not np.isfinite(start+end) or start>=end or not mask.any() or np.any(used&mask):
            raise ValueError('Invalid/overlapping diagnostic supports')
        if not np.array_equal(a['times_seconds'],t[mask]):raise ValueError('Changed strict-law time grid')
        cents=np.asarray(a['sign_preserving_cents'])
        if cents.shape!=t[mask].shape or not np.isfinite(cents).all():raise ValueError('Invalid strict control law')
        result[mask]=cents;used|=mask
    return result


def residual_summary(delta, law, mask):
    """Diagnostic residual, not a new audit: raw pitch change minus center law."""
    delta=np.asarray(delta);law=np.asarray(law);mask=np.asarray(mask)
    if (delta.ndim!=1 or law.shape!=delta.shape or mask.shape!=delta.shape or mask.dtype!=bool
            or not np.isfinite(law).all()):raise ValueError('Invalid residual diagnostic grid')
    valid=mask&np.isfinite(delta)
    if not valid.any():raise ValueError('No valid diagnostic frames')
    residual=delta-law
    positions=np.flatnonzero(valid);i=int(positions[np.argmax(abs(residual[valid]))])
    return dict(valid_frames=int(valid.sum()),max_abs_center_law_residual_cents=float(abs(residual[i])),
                residual_peak_frame=i,delta_at_residual_peak_cents=float(delta[i]),
                center_law_at_residual_peak_cents=float(law[i]),signed_residual_at_peak_cents=float(residual[i])),residual


def native_context(features, index):
    """Show the nearest native frame actually selected by analysis.extract."""
    t=features['times_seconds'][index];nt=features['native_pitch_times_seconds']
    j=int(np.clip(np.searchsorted(nt,t),0,len(nt)-1));previous=max(j-1,0)
    if abs(nt[previous]-t)<abs(nt[j]-t):j=previous
    return dict(native_frame=j,native_time_seconds=float(nt[j]),
                native_hz=float(features['native_pitch_hz'][j]),
                native_confidence=float(features['native_pitch_confidence'][j]),
                aligned_hz=float(features['f0_hz'][index]))


def run():
    import librosa
    import torch
    component=Path(__file__).resolve().parents[2]
    path=component/SOURCE;prior=json.loads(path.read_text(encoding='utf-8'))
    anchor_path=component/ANCHOR;anchor=json.loads(anchor_path.read_text(encoding='utf-8'))
    baseline_path=component/BASELINE;baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    if prior['run_status']!='complete' or prior['adopted'] or prior['ordinary_vocal_renderer']:
        raise ValueError('Wrong frozen source-control evidence')
    tracked=dict(prior['tracked_inputs']);tracked[str(path)]=checksum(path)
    for item in prior['artifacts']:tracked[item['path']]=item['sha256']
    for p in (Path(__file__),Path(__file__).parent/'framewise_diagnostics.py',anchor_path,baseline_path):
        tracked[str(p)]=checksum(p)
    for p,h in tracked.items():
        if checksum(p)!=h:raise ValueError('Changed frozen input: '+p)
    root=component/'runs/known_control_diagnostics'/uuid.uuid4().hex;root.mkdir(parents=True)
    snapshots=root/'source_code_at_run';snapshots.mkdir()
    for p in tracked:
        if Path(p).suffix=='.py':(snapshots/Path(p).name).write_bytes(Path(p).read_bytes())
    write_json(root/'declaration.json',DECLARATION)
    record=dict(run_status='running',version=VERSION,data_status='exploratory_not_research',declaration=DECLARATION,
                declaration_sha256=checksum(root/'declaration.json'),tracked_inputs=tracked,artifacts=[],cases=[],
                withheld=[dict(profile=c['profile'],curve=c['curve'],candidate_path=None,outcome=c['outcome'])
                          for c in prior['cases'] if not c['rendered']],
                cpu_threads=torch.get_num_threads(),new_candidate_wavs=0,adopted=False,
                training_ready_examples=0,accepted_prepared_packages=0)
    def save():write_json(root/'evidence.json',record)
    save();print('KNOWN_CONTROL_DIAGNOSTICS_RUN:',root,flush=True)
    try:
        for case in (c for c in prior['cases'] if c['rendered']):
            print('PROFILE:',case['profile'],flush=True)
            profile=next(p for p in baseline['profiles'] if p['name']==case['profile'])
            source,sr=sf.read(case['source_path'],dtype='float32');st,f0,amp=constructed(profile)
            if sr!=22050 or not np.array_equal(source,signal(f0,amp)):raise ValueError('Source construction changed')
            before,analyzer=extract(source,sr,component/'checkpoints')
            expected=next(c for c in baseline['cases'] if c['profile']==case['profile'])['source_analyzer']
            if analyzer!=expected:raise ValueError('Analyzer/checkpoint provenance changed')
            times=before['times_seconds'];y0=independent_pitch(source,sr)
            refs={2048:y0,1024:librosa.yin(source,sr=sr,fmin=50.,fmax=1500.,frame_length=1024,hop_length=441)}
            reliable=before['voiced_mask']&np.isfinite(y0)&(y0>0)
            reliable[reliable]&=abs(1200*np.log2(y0[reliable]/before['f0_hz'][reliable]))<=AuditPolicy().tracker_agreement_cents
            if profile['kind']=='rest':reliable&=(times<2.5)|(times>=4.5)
            requested=requested_curve(times,case['intervals'],case['curve'])
            model=next(c for c in anchor['cases'] if (c['profile'],c['curve'])==(case['profile'],case['curve']))
            supports=[]
            for support in model['supports']:
                with np.load(support['arrays']) as a:arrays={k:a[k] for k in a.files}
                supports.append((support['start_seconds'],support['end_seconds'],arrays))
            sample_law=strict_sample_law(st,supports);center_law=np.interp(times,st,sample_law)
            mask=transition_mask(times,case['intervals']);out=dict(times=times,requested=requested,
                center_law_cents=center_law,transition_mask=mask,source_center_hz=np.interp(times,st,f0),
                source_pesto_hz=before['f0_hz'],source_yin_2048=y0,source_yin_1024=refs[1024])
            result=dict(profile=case['profile'],curve=case['curve'],methods={})
            for name,key,audit_key in (('baseline','baseline_candidate_path','baseline_audit'),('control','candidate_path','audit')):
                wave,rate=sf.read(case[key],dtype='float32')
                if rate!=sr or wave.shape!=source.shape:raise ValueError('Saved WAV geometry changed')
                after,after_analyzer=extract(wave,sr,component/'checkpoints')
                if after_analyzer!=analyzer:raise ValueError('Candidate analyzer changed')
                y1=independent_pitch(wave,sr)
                audit=json.loads(json.dumps(audit_curve(before,after,y0,y1,reliable,case['intervals'],requested)))
                if audit!=case[audit_key]:raise ValueError('Guarded audit replay changed: '+case['profile']+'/'+name)
                rows,detail=direction_details(before,after,y0,y1,reliable,case['intervals'],requested)
                if len(rows)!=audit['coarse_wrong_direction_frames']:raise ValueError('Opposing-frame mask mismatch')
                for r in rows:
                    i=r['frame'];r.update(source_center_hz=float(out['source_center_hz'][i]),
                        source_native=native_context(before,i),output_native=native_context(after,i))
                    if name=='control':r['known_control_center_law_cents']=float(center_law[i])
                out.update({name+'_'+k:v for k,v in detail.items()})
                measured=dict(audit_reproduced_exactly=True,audit=audit,opposing_frames=rows,transitions={})
                for w,reference in refs.items():
                    measured_yin=y1 if w==2048 else librosa.yin(wave,sr=sr,fmin=50.,fmax=1500.,frame_length=w,hop_length=441)
                    peak,delta=relative_peak(reference,measured_yin,mask)
                    if peak!=case['transition_peaks'][str(w)][name]:raise ValueError('Transition peak replay changed')
                    with np.load(case['arrays']) as a:old_delta=a[name+'_transition_delta_'+str(w)]
                    if not np.array_equal(delta,old_delta,equal_nan=True):raise ValueError('Saved transition deltas changed')
                    info=dict(original_change_peak=peak,raw_peak_time_seconds=float(times[peak['peak_frame']]))
                    out[name+'_yin_'+str(w)]=measured_yin;out[name+'_transition_delta_'+str(w)]=delta
                    if name=='control':
                        metrics,residual=residual_summary(delta,center_law,mask)
                        i=peak['peak_frame'];j=metrics['residual_peak_frame']
                        metrics.update(residual_peak_time_seconds=float(times[j]),
                            center_law_at_raw_peak_cents=float(center_law[i]),
                            signed_residual_at_raw_peak_cents=float(residual[i]))
                        # Algebraic check with pointwise construction frequencies.
                        expected_source=out['source_center_hz'];expected_output=expected_source*2**(center_law/1200.)
                        e0,_=cents_difference(reference,expected_source,np.ones(len(times),dtype=bool))
                        e1,_=cents_difference(measured_yin,expected_output,np.ones(len(times),dtype=bool))
                        valid=mask&np.isfinite(delta)
                        if not np.allclose(residual[valid],(e1-e0)[valid],rtol=0,atol=1e-9):raise ValueError('Residual decomposition mismatch')
                        out['control_center_law_residual_'+str(w)]=residual
                        out['source_center_frequency_residual_'+str(w)]=e0
                        out['control_center_frequency_residual_'+str(w)]=e1
                        info['center_law_diagnostic']=metrics
                    measured['transitions'][str(w)]=info
                result['methods'][name]=measured
            array_path=root/(case['profile']+'_'+case['curve']+'.npz');np.savez_compressed(array_path,**out)
            record['artifacts'].append(dict(path=str(array_path),sha256=checksum(array_path)))
            result['arrays']=str(array_path);record['cases'].append(result);save()
        cases=record['cases'];rows=[r for c in cases for r in c['methods']['control']['opposing_frames']]
        record['summary']=dict(cases=len(cases),withheld=len(record['withheld']),audit_dictionaries_replayed=2*len(cases),
            transition_peaks_replayed=4*len(cases),control_passes=sum(c['methods']['control']['audit']['passed'] for c in cases),
            baseline_passes=sum(c['methods']['baseline']['audit']['passed'] for c in cases),opposing_control_frames=len(rows),
            max_opposing_control_abs_cents=max((abs(r['coarse_cents']) for r in rows),default=0.),
            max_opposing_control_float64_abs_cents=max((abs(r['float64_ratio_coarse_cents']) for r in rows),default=0.),
            center_law_diagnostics={str(w):dict(
                max_abs_residual_cents=max(c['methods']['control']['transitions'][str(w)]['center_law_diagnostic']['max_abs_center_law_residual_cents'] for c in cases),
                per_case=[dict(profile=c['profile'],**c['methods']['control']['transitions'][str(w)]['center_law_diagnostic']) for c in cases]) for w in (2048,1024)})
        if any(checksum(p)!=h for p,h in tracked.items()) or checksum(root/'declaration.json')!=record['declaration_sha256']:
            raise ValueError('Frozen diagnostic inputs changed')
        record['run_status']='complete';save();print(json.dumps(record['summary'],indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
