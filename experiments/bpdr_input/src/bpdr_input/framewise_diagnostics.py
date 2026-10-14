"""Frozen replay and diagnostic contrasts; does not change any acceptance policy."""
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from .analysis import extract
from .data_builder import write_json
from .framewise_audit import audit_curve
from .framewise_fixtures import constructed, signal, requested_curve
from .framewise_renderer import RendererPolicy
from .interventions import AuditPolicy, cents_difference, envelope, independent_pitch
from .pipeline import checksum
from .notebook_evidence import component_path


SOURCE_RUN = 'runs/framewise_fixtures/1ae10a68d8c94dcc89d18c45719cef6f/evidence.json'
DECLARATION = dict(
    selection='All 32 original constructed cases; no outcome-based exclusions',
    diagnostic_magnitude_bins_cents=[.001, 1., 10.],
    bins_meaning='Descriptive sizes only; no epsilon, new pass criterion or acceptance relaxation',
    contrasts=['saved local candidate', 'private full shifted PSOLA',
               'private unchanged-tier PSOLA', 'analytic shifted-contour oracle'],
    transition_window='[start-.10,start+.18) union (end-.18,end+.10], original timestamps',
    yin_window_samples=[2048, 1024],
    window_meaning='2048 is the existing auditor; 1024 is a fixed sensitivity diagnostic, never substituted',
    audible_quality='Not measured; estimator peaks are not instantaneous-pitch ground truth',
    adoption=False, accepted_prepared_packages=0, training_ready_examples=0)


def transition_mask(times, intervals):
    mask=np.zeros(len(times), dtype=bool)
    for start,end in intervals:
        mask|=((times>=start-.10)&(times<start+.18))|((times>end-.18)&(times<=end+.10))
    return mask


def direction_details(before, after, yin_before, yin_after, reliable, intervals, curve):
    """Reproduce the audit mask and retain exact frequency values at opposing frames."""
    times=before['times_seconds'];plateau=np.zeros(len(times),dtype=bool)
    policy=AuditPolicy()
    for start,end in intervals:
        plateau|=(times>=start+policy.ramp_seconds+policy.analysis_guard_seconds)&(times<=end-policy.ramp_seconds-policy.analysis_guard_seconds)
    coarse,pvalid=cents_difference(after['f0_hz'],before['f0_hz'],before['voiced_mask']&after['voiced_mask'])
    fine,yvalid=cents_difference(yin_after,yin_before,reliable)
    common=plateau&reliable&pvalid&yvalid
    direction=common&(abs(curve)>=10.)
    opposing=direction&(coarse*curve<0.)
    coarse64,_=cents_difference(after['f0_hz'].astype(np.float64),before['f0_hz'].astype(np.float64),pvalid)
    rows=[]
    for i in np.flatnonzero(opposing):
        size=abs(float(coarse[i]));thresholds=DECLARATION['diagnostic_magnitude_bins_cents']
        rows.append(dict(frame=int(i),time_seconds=float(times[i]),requested_cents=float(curve[i]),
            coarse_cents=float(coarse[i]),float64_ratio_coarse_cents=float(coarse64[i]),fine_cents=float(fine[i]),
            before_hz=float(before['f0_hz'][i]),after_hz=float(after['f0_hz'][i]),
            frequency_difference_hz=float(after['f0_hz'][i])-float(before['f0_hz'][i]),
            before_confidence=float(before['confidence'][i]),after_confidence=float(after['confidence'][i]),
            size_bin=('<=.001' if size<=thresholds[0] else '.001-1' if size<=thresholds[1] else '1-10' if size<=thresholds[2] else '>10')))
    arrays=dict(times=times,requested=curve,coarse_delta=coarse,coarse_delta_float64=coarse64,
        fine_delta=fine,common=common,direction=direction,opposing=opposing,
        before_hz=before['f0_hz'],after_hz=after['f0_hz'],
        before_confidence=before['confidence'],after_confidence=after['confidence'])
    return rows,arrays


def private_resynthesis(audio, sr, times, curve, policy=RendererPolicy()):
    """Diagnostic full output only. Reconstructed local output MUST match saved samples.

    This intentionally exposes the existing algorithm's private contrast without
    altering the original renderer or routing whole resynthesis into a transaction.
    Zero curve here measures backend artifacts; real zero edits still bypass it.
    """
    import parselmouth
    from parselmouth.praat import call
    sound=parselmouth.Sound(audio.astype(np.float64),sampling_frequency=sr,start_time=0.)
    manipulation=call(sound,'To Manipulation',policy.time_step_seconds,policy.pitch_floor_hz,policy.pitch_ceiling_hz)
    original=call(manipulation,'Extract pitch tier')
    tier=call('Create PitchTier','bpdr_candidate',sound.xmin,sound.xmax)
    for index in range(1,int(call(original,'Get number of points'))+1):
        t=float(call(original,'Get time from index',index));f=float(call(original,'Get value at index',index))
        call(tier,'Add point',t,f*2**(float(np.interp(t,times,curve,left=0.,right=0.))/1200.))
    call([tier,manipulation],'Replace pitch tier')
    output=call(manipulation,'Get resynthesis (overlap-add)')
    if (output.values.shape!=(1,len(audio)) or output.sampling_frequency!=sr or
            (output.xmin,output.xmax,output.x1)!=(sound.xmin,sound.xmax,sound.x1)):
        raise ValueError('Diagnostic native output geometry changed')
    return output.values[0]


def relative_peak(reference, candidate, mask):
    delta,valid=cents_difference(candidate,reference,mask)
    if not valid.any(): raise ValueError('No valid transition frames')
    positions=np.flatnonzero(valid);i=int(positions[np.argmax(abs(delta[valid]))])
    return dict(valid_frames=int(valid.sum()),peak_abs_cents=float(abs(delta[i])),
                signed_cents_at_peak=float(delta[i]),peak_frame=i),delta


def run():
    import librosa
    import parselmouth
    import torch
    component=Path(__file__).resolve().parents[2]
    source_path=component/SOURCE_RUN;source=json.loads(source_path.read_text(encoding='utf-8'))
    if source['run_status']!='complete' or source['adopted']: raise ValueError('Wrong source run')
    if source['renderer_policy']!=asdict(RendererPolicy()):raise ValueError('Renderer policy changed')
    if source['base_audit_limits']!=json.loads(json.dumps(asdict(AuditPolicy()))):raise ValueError('Audit limits changed')
    tracked={str(source_path):checksum(source_path),str(source_path.parent/'transition_diagnostic.json'):checksum(source_path.parent/'transition_diagnostic.json')}
    for name,digest in source['code_sha256'].items():
        path=Path(__file__).parent/(name+'.py')
        if checksum(path)!=digest:raise ValueError('Original measured code changed: '+name)
        tracked[str(path)]=digest
    for item in source['artifacts']:
        path=component_path(component,item['path'])
        if checksum(path)!=item['sha256']:raise ValueError('Original waveform changed')
        tracked[str(path)]=item['sha256']
    if checksum(parselmouth.__file__)!=source['backend_sha256']:raise ValueError('Native backend changed')
    root=component/'runs/framewise_diagnostics'/uuid.uuid4().hex;root.mkdir(parents=True)
    own_path=Path(__file__);own_hash=checksum(own_path)
    (root/own_path.name).write_bytes(own_path.read_bytes())
    record=dict(run_status='running',scope='Gate F diagnostic only',data_status='exploratory_not_research',
        declaration=DECLARATION,source_evidence=str(source_path),source_evidence_sha256=checksum(source_path),
        tracked_inputs=tracked,diagnostic_code_sha256=own_hash,backend_sha256=source['backend_sha256'],
        cpu_threads=torch.get_num_threads(),artifacts=[],cases=[],controls=[],adopted=False,
        accepted_prepared_packages=0,training_ready_examples=0)
    def save():write_json(root/'evidence.json',record)
    def artifact(path):record['artifacts'].append(dict(path=str(path),sha256=checksum(path)))
    save();print('DIAGNOSTIC_RUN:',root,flush=True)
    try:
        for profile in source['profiles']:
            print('PROFILE:',profile['name'],flush=True)
            cases=[c for c in source['cases'] if c['profile']==profile['name']]
            audio,sr=sf.read(cases[0]['source_path'],dtype='float32')
            sample_times,f0,amplitude=constructed(profile)
            if sr!=22050 or not np.array_equal(audio,signal(f0,amplitude)):raise ValueError('Constructed source mismatch')
            before,analyzer=extract(audio,sr,component/'checkpoints');times=before['times_seconds']
            if analyzer!=cases[0]['source_analyzer']:raise ValueError('Source analyzer provenance changed')
            yin=independent_pitch(audio,sr)
            reliable=before['voiced_mask']&np.isfinite(yin)&(yin>0)
            reliable[reliable]&=abs(1200*np.log2(yin[reliable]/before['f0_hz'][reliable]))<=AuditPolicy().tracker_agreement_cents
            if profile['kind']=='rest':reliable&=(times<2.5)|(times>=4.5)
            nochange=private_resynthesis(audio,sr,times,np.zeros(len(times))).astype(np.float32)
            refs={2048:yin,1024:librosa.yin(audio,sr=sr,fmin=50.,fmax=1500.,frame_length=1024,hop_length=441)}
            unchanged={w:librosa.yin(nochange,sr=sr,fmin=50.,fmax=1500.,frame_length=w,hop_length=441) for w in refs}
            mask=transition_mask(times,cases[0]['intervals'])
            record['controls'].append(dict(profile=profile['name'],actual_zero_edit_is_exact_bypass=True,
                private_nochange_samples_differ=int(np.count_nonzero(nochange!=audio)),
                peaks={str(w):relative_peak(refs[w],unchanged[w],mask)[0] for w in refs}))
            for case in cases:
                candidate,rate=sf.read(case['candidate_path'],dtype='float32')
                if rate!=sr:raise ValueError('Candidate rate changed')
                curve=requested_curve(times,case['intervals'],case['curve'])
                after,_=extract(candidate,sr,component/'checkpoints');candidate_yin=independent_pitch(candidate,sr)
                audit=audit_curve(before,after,yin,candidate_yin,reliable,case['intervals'],curve)
                if json.loads(json.dumps(audit))!=case['audit']:raise ValueError('Original audit did not replay exactly: '+profile['name']+'/'+case['curve'])
                opposing,arrays=direction_details(before,after,yin,candidate_yin,reliable,case['intervals'],curve)
                if len(opposing)!=audit['coarse_wrong_direction_frames']:raise ValueError('Direction mask disagrees with audit')
                full=private_resynthesis(audio,sr,times,curve)
                blend=np.zeros(len(audio))
                for support in case['intervals']:blend=np.maximum(blend,envelope(sample_times,support,RendererPolicy().ramp_seconds))
                reproduced=(audio*(1-blend)+full*blend).astype(np.float32)
                if not np.array_equal(reproduced,candidate):raise ValueError('Private contrast does not reproduce saved local samples')
                oracle=signal(f0*2**(np.interp(sample_times,times,curve)/1200.),amplitude)
                contrasts=dict(local=candidate,full=full.astype(np.float32),oracle=oracle)
                peaks={};arrays['transition_mask']=mask
                for w in refs:
                    peaks[str(w)]={}
                    for name,wave in contrasts.items():
                        measured=candidate_yin if name=='local' and w==2048 else librosa.yin(wave,sr=sr,fmin=50.,fmax=1500.,frame_length=w,hop_length=441)
                        peaks[str(w)][name],arrays[f'transition_delta_{name}_{w}']=relative_peak(refs[w],measured,mask)
                    arrays[f'transition_delta_nochange_{w}']=relative_peak(refs[w],unchanged[w],mask)[1]
                path=root/(profile['name']+'_'+case['curve']+'.npz');np.savez_compressed(path,**arrays);artifact(path)
                record['cases'].append(dict(profile=profile['name'],curve=case['curve'],audit_reproduced_exactly=True,
                    passed=audit['passed'],reasons=audit['reasons'],opposing_frames=opposing,
                    local_waveform_reproduced_exactly=True,transition_peaks=peaks,arrays=str(path)))
                save()
        prior=json.loads((source_path.parent/'transition_diagnostic.json').read_text(encoding='utf-8'))
        # Each old peak must replay, independently of the additional contrasts.
        old=prior['cases'] if 'cases' in prior else prior['records']
        for c,p in zip(record['cases'],old):
            if (c['profile'],c['curve'])!=(p['profile'],p['curve']) or not np.isclose(c['transition_peaks']['2048']['local']['peak_abs_cents'],p['yin_peak_abs_relative_cents'],rtol=0,atol=1e-9):
                raise ValueError('Transition diagnostic replay mismatch')
        rows=[r for c in record['cases'] for r in c['opposing_frames']]
        record['summary']=dict(cases=len(record['cases']),unchanged_audit_passes=sum(c['passed'] for c in record['cases']),
            rejected_cases=sum(not c['passed'] for c in record['cases']),opposing_frames=len(rows),
            magnitude_bins={label:sum(r['size_bin']==label for r in rows) for label in ('<=.001','.001-1','1-10','>10')},
            max_opposing_abs_cents=max(abs(r['coarse_cents']) for r in rows),
            max_opposing_float64_abs_cents=max(abs(r['float64_ratio_coarse_cents']) for r in rows),
            transition_exceeds30={str(w):{name:sum(c['transition_peaks'][str(w)][name]['peak_abs_cents']>30 for c in record['cases']) for name in contrasts} for w in refs})
        if any(checksum(p)!=h for p,h in tracked.items()) or checksum(own_path)!=own_hash or checksum(parselmouth.__file__)!=record['backend_sha256']:
            raise ValueError('Source changed during diagnostic')
        record['run_status']='complete';save();print(json.dumps(record['summary'],indent=2),flush=True)
    except BaseException as exc:
        record.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return root


if __name__=='__main__':run()
