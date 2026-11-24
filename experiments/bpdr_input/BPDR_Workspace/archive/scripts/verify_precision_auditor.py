"""Verify complete precision-auditor comparison, including independent encoded-WAV audit."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import evaluate_precision_auditor as study
from bpdr_input.audio.analysis import _pitch_model
from bpdr_input.repair.framewise_renderer import render_candidate
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.rule_correction import fine_hz
from bpdr_input.repair.repair_controller import waveform_hash
from verify_frozen_audio_studies import outputs


def main():
    torch.set_num_threads(2);d=study.declare();ex=study.diagnostic.upstream.study.ex;root=study.ROOT;folder=root/'arm/training/median11';baseline=Path(d['baseline_folder'])
    rows=ex.read(d['manifest']);pairs=study.diagnostic.upstream.study.training.held.prepared(rows)
    old=ex.read(baseline/'accuracy.json');new=ex.read(folder/'accuracy.json')
    assert len(new['rows'])==len(old['rows'])==200 and sum('excluded' not in r for r in rows)==108
    replayed_arrays=0
    with np.load(baseline/'predictions.npz') as a,np.load(folder/'predictions.npz') as b:
        assert set(a.files)==set(b.files)
        for key in a.files:np.testing.assert_array_equal(a[key],b[key]);replayed_arrays+=1
    controls=ex.read(folder/'original_controls.json');original=ex.read(baseline/'original_controls.json')
    for a,b in zip(original,controls,strict=True):
        assert a['source_sha256']==b['source_sha256'] and a['raw_original_abs_cents']==b['raw_original_abs_cents'] and b['delivered_identity']
        assert a['transaction']['plan']==b['transaction']['plan']
    auth=outputs(folder,Path(d['manifest']),ex.read(study.diagnostic.upstream.study.ROOT/'anchor_sources.json'))
    model=_pitch_model(str((ex.COMPONENT_ROOT/'checkpoints').resolve()));cache={}
    def features(audio,times):
        key=waveform_hash(audio)
        if key not in cache:
            with torch.inference_mode():_,conf,_,act=model(torch.as_tensor(audio,dtype=torch.float32),22050,convert_to_freq=True,return_activations=True)
            a=act.double().reshape(-1,384);center=a.argmax(-1)[:,None];idx=(center+torch.arange(-2,3)).clamp(0,383)
            w=torch.gather(a,1,idx);pitch=(w*(idx.double()/3)).sum(-1)/w.sum(-1);native=(440*torch.pow(torch.tensor(2.,dtype=torch.float64),(pitch-69)/12)).numpy()
            confidence=conf.numpy().reshape(-1);nt=np.arange(len(native))*(round(model.hop_size*22050/1000)/22050)
            index=np.clip(np.searchsorted(nt,times),0,len(nt)-1);prev=np.maximum(index-1,0);index=np.where(abs(nt[prev]-times)<abs(nt[index]-times),prev,index)
            voiced=(confidence[index]>=.8)&(native[index]>0)
            cache[key]=(dict(times_seconds=times,f0_hz=np.where(voiced,native[index],np.nan),voiced_mask=voiced),fine_hz(audio,22050,times))
        return cache[key]
    def audit(a,b,times,reliable,intervals,requested):
        before,yb=features(a,times);after,ya=features(b,times)
        return audit_curve(before,after,yb,ya,reliable,intervals,requested)
    def compare(expected,actual):
        assert expected['passed']==actual['passed'] and expected['reasons']==actual['reasons']
        for key in ('common_frames','fine_wrong_direction_frames','coarse_wrong_direction_frames','direction_checked_frames','outside_frames'):assert expected[key]==actual[key],key
        for name in ('fine','coarse'):
            assert expected[name]['frames']==actual[name]['frames']
            for key in ('median_error_cents','p90_error_cents'):np.testing.assert_allclose(expected[name][key],actual[name][key],rtol=0,atol=1e-8)
    files={ex.sha(p):p for p in (folder/'audio').rglob('*_candidate.wav')};candidate_count=encoded_count=0
    with np.load(folder/'predictions.npz') as z:
        for i,(p,row,prior) in enumerate(zip(pairs,new['rows'],old['rows'],strict=True)):
            if 'excluded' in p:assert row['excluded']==prior['excluded'];continue
            for view in ('A','C'):
                trace=row['trace'][view];assert trace['permission']==prior['trace'][view]['permission']
                plan=trace['permission']
                if not plan['permitted'] or plan['identity']:continue
                audio=p['audio_'+view];times=p[view]['features']['times_seconds'];curve=z[f'{i}_requested_{view}'];reliable=p[view]['reliable']
                candidate,_=render_candidate(audio,22050,times,curve,plan['intervals_seconds'])
                actual=audit(audio,candidate,times,reliable,plan['intervals_seconds'],curve)
                compare(trace['transaction']['post_render_audit'],actual);candidate_count+=1
                if trace['outcome']=='VERIFIED_FIXTURE_CANDIDATE':
                    accepted=trace['transaction']['delivered_wav_audit'];wave,sr=sf.read(files[accepted['sha256']],dtype='float32');assert sr==22050
                    realized=audit(audio,wave,times,reliable,plan['intervals_seconds'],curve);compare(accepted['acoustic_audit'],realized);assert realized['passed'];encoded_count+=1
                print('PRECISION_PIPELINE_CASE_VERIFIED',i,view,trace['outcome'],flush=True)
    evidence=ex.read(folder/'evidence.json');ref=ex.read(baseline/'evidence.json');choice=ex.read(root/'locked_selection.json')
    chosen=bool(old['prediction_gate_passed'] and evidence['gates']['original_controls_all_identity'] and evidence['summary']['delivered_after_mae_cents']<ref['summary']['delivered_after_mae_cents'])
    assert choice['selected_training_side']==chosen and not choice['demo_activated']
    protected=ex.read(root.parent/'2026-10-28_v7_guard/declaration.json')['protected_files']
    for path,digest in protected.items():assert ex.sha(path)==digest
    assert not (root.parent/'active_model.json').exists();study.declare()
    receipt=dict(**auth,all_prediction_curve_arrays_identical=replayed_arrays,permission_decisions_identical=216,independent_float64_candidate_audits=candidate_count,independent_float64_encoded_audits=encoded_count,all50_original_raw_predictions_and_plans_exact=True,selected_training_side=chosen,protected_files_unchanged=len(protected),evidence_sha256=ex.sha(folder/'evidence.json'),selection_sha256=ex.sha(root/'locked_selection.json'),verifier_sha256=ex.sha(Path(__file__)),activation_absent=True,research_validation_passed=False)
    ex.write(root/'integrity_receipt.json',receipt);print('PRECISION_PIPELINE_VERIFIED',receipt,flush=True)

if __name__=='__main__':main()
