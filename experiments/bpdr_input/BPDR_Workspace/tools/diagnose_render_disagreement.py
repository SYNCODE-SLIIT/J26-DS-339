"""Reconstruct all frozen nonidentity candidates; explain pitch-audit disagreements."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import csv
import numpy as np
import torch
import train_nearby_zero_transformer as study
from bpdr_input.audio.analysis import extract
from bpdr_input.repair.framewise_renderer import render_candidate
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.interventions import AuditPolicy,cents_difference
from bpdr_input.repair.rule_correction import fine_hz
from bpdr_input.repair.repair_controller import decision_hash,waveform_hash

ROOT=study.ROOT.parent/'2026-11-01_render_disagreement'


def direction_details(before,after,fine_before,fine_after,reliable,intervals,requested):
    policy=AuditPolicy();times=before['times_seconds']
    plateau=np.zeros(len(times),bool)
    for lo,hi in intervals:
        plateau|=(times>=lo+policy.ramp_seconds+policy.analysis_guard_seconds)&(times<=hi-policy.ramp_seconds-policy.analysis_guard_seconds)
    coarse,cv=cents_difference(after['f0_hz'],before['f0_hz'],before['voiced_mask']&after['voiced_mask'])
    fine,fv=cents_difference(fine_after,fine_before,reliable)
    checked=plateau&reliable&cv&fv&(abs(requested)>=10.)
    opposing=checked&(coarse*requested<0)
    return checked,opposing,fine,coarse


def declare():
    ex=study.ex;d=study.declare(); ROOT.mkdir(exist_ok=True)
    baseline=Path(d['baseline_evidence']).parent
    arms={'reference':baseline,'nearby_zero':study.ROOT/'arm/training/median11'}
    if not (ROOT/'declaration.json').exists():
        assert ex.read(study.ROOT/'integrity_receipt.json')['sampling_folds_verified']==4
        inputs=dict(d['input_hashes'])
        for folder in arms.values():
            for name in ('accuracy.json','predictions.npz','evidence.json'):
                inputs[str(folder/name)]=ex.sha(folder/name)
        sources=dict(d['source_hashes'])
        sources[str(Path(__file__).resolve())]=ex.sha(Path(__file__))
        sources[str(Path(__file__).with_name('verify_render_disagreement.py').resolve())]=ex.sha(Path(__file__).with_name('verify_render_disagreement.py'))
        ex.write(ROOT/'declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_hashes=sources,input_hashes=inputs,arms={k:str(v) for k,v in arms.items()},manifest=str(study.ROOT/'train_pairs.json'),manifest_sha256=ex.sha(study.ROOT/'train_pairs.json'),
            protocol='Inspect every200 attempted cases per arm,108 scoreable/92 excluded. Re-render every permitted nonidentity C candidate (expected30 per arm) using its exact frozen requested curve and permission support. Reproduce existing post-render audit exactly. Save private diagnostic candidate arrays and before/after native/aligned coarse,fine,confidence,reliable/requested/direction masks and per-opposing-frame numerical measurements. Include successful candidates as controls. No re-fitting, reference/known labels in renderer, per-case routing, threshold changes or accepted rejected audio.',
            selection_eligible=False,diagnostic_only=True,research_validation_passed=False,demo_activated=False))
    locked=ex.read(ROOT/'declaration.json')
    for kind in ('source_hashes','input_hashes'):
        for p,h in locked[kind].items(): assert ex.sha(p)==h,p
    assert ex.sha(locked['manifest'])==locked['manifest_sha256']
    return locked


def main():
    torch.set_num_threads(2);d=declare();ex=study.ex
    if (ROOT/'results.json').exists():raise ValueError('Retain completed diagnostic')
    rows=ex.read(d['manifest']);pairs=study.training.held.prepared(rows)
    reports={};frame_rows=[]
    for arm,folder in d['arms'].items():
        folder=Path(folder);accuracy=ex.read(folder/'accuracy.json');outcomes=Counter();cases=[]
        with np.load(folder/'predictions.npz') as z:
            for i,(p,row) in enumerate(zip(pairs,accuracy['rows'],strict=True)):
                if 'excluded' in p:outcomes['excluded']+=1;continue
                trace=row['trace']['C'];plan=trace['permission'];outcomes[trace['outcome']]+=1
                assert decision_hash(plan)==plan['decision_sha256']
                if not plan['permitted'] or plan['identity']:continue
                out=ROOT/arm/f'case_{i:03d}';out.mkdir(parents=True,exist_ok=True)
                if (out/'receipt.json').exists():
                    r=ex.read(out/'receipt.json');assert ex.sha(out/'tracks.npz')==r['tracks_sha256'];cases.append(r);frame_rows.extend(r['opposing_frames']);continue
                audio=p['audio_C'];before=p['C']['features'];reliable=p['C']['reliable'];times=before['times_seconds'];requested=z[f'{i}_requested_C'].copy()
                assert waveform_hash(audio)==plan['source_sha256']
                candidate,render=render_candidate(audio,22050,times,requested,plan['intervals_seconds'])
                after,analyzer=extract(candidate,22050,study.ex.COMPONENT_ROOT/'checkpoints')
                fb=fine_hz(audio,22050,times);fa=fine_hz(candidate,22050,times)
                audit=audit_curve(before,after,fb,fa,reliable,plan['intervals_seconds'],requested)
                old=trace['transaction']['post_render_audit'];assert audit==old,(arm,i,audit,old)
                if audit['passed']:assert waveform_hash(candidate)==trace['transaction']['candidate_sha256']
                checked,opposing,fine,coarse=direction_details(before,after,fb,fa,reliable,plan['intervals_seconds'],requested)
                assert int(opposing.sum())==audit['coarse_wrong_direction_frames']
                samples=np.arange(len(audio))/22050;inside=np.zeros(len(audio),bool)
                for lo,hi in plan['intervals_seconds']:inside|=(samples>=lo)&(samples<hi)
                assert np.array_equal(candidate[~inside],audio[~inside])
                arrays=dict(candidate_audio=candidate,times=times,requested=requested,reliable=reliable,checked=checked,opposing=opposing,fine_before=fb,fine_after=fa,fine_shift=fine,coarse_shift=coarse)
                for view,features in [('before',before),('after',after)]:
                    for key in ('f0_hz','confidence','voiced_mask','native_pitch_times_seconds','native_pitch_hz','native_pitch_confidence'):
                        arrays[view+'_'+key]=features[key]
                np.savez_compressed(out/'tracks.npz',**arrays)
                details=[]
                for j in np.flatnonzero(opposing):
                    item=dict(arm=arm,case_index=i,group=row['group'],frame=int(j),time_seconds=float(times[j]),requested_cents=float(requested[j]),fine_shift_cents=float(fine[j]),coarse_shift_cents=float(coarse[j]),coarse_before_hz=float(before['f0_hz'][j]),coarse_after_hz=float(after['f0_hz'][j]),confidence_before=float(before['confidence'][j]),confidence_after=float(after['confidence'][j]))
                    details.append(item)
                r=dict(case_index=i,arm=arm,group=row['group'],existing_outcome=trace['outcome'],audit=audit,intervals_seconds=plan['intervals_seconds'],plan_sha256=plan['decision_sha256'],candidate_sha256=waveform_hash(candidate),tracks_sha256=ex.sha(out/'tracks.npz'),opposing_frames=details,analyzer=analyzer,diagnostic_only=True,production_eligible=False)
                ex.write(out/'receipt.json',r);cases.append(r);frame_rows.extend(details)
                print('DISAGREEMENT_CASE',arm,i,'opposing',len(details),flush=True)
        reports[arm]=dict(attempted=len(rows),scoreable=sum('excluded' not in r for r in rows),outcomes=dict(outcomes),rendered_candidates=len(cases),opposing_frames=sum(len(r['opposing_frames']) for r in cases),reproduced_audits=len(cases),cases=cases)
    if frame_rows:
        with (ROOT/'opposing_frames.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(frame_rows[0]));writer.writeheader();writer.writerows(frame_rows)
    protected=ex.read(ROOT.parent/'2026-10-28_v7_guard/declaration.json')['protected_files']
    for p,h in protected.items():assert ex.sha(p)==h
    declare();result=dict(arms=reports,declaration_sha256=ex.sha(ROOT/'declaration.json'),protected_files_unchanged=len(protected),selection_eligible=False,diagnostic_only=True,research_validation_passed=False,demo_activated=False)
    ex.write(ROOT/'results.json',result)
    print('RENDER_DISAGREEMENT_COMPLETE',{k:(v['rendered_candidates'],v['opposing_frames']) for k,v in reports.items()},flush=True)

if __name__=='__main__':main()
