"""Independent full-fit lineage and known-development waveform/score verification."""
import argparse,json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import qualify_centered_development as study
from bpdr_input.audio.analysis import _pitch_model
from bpdr_input.repair.framewise_renderer import render_candidate
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.rule_correction import fine_hz,fine_pitch
from bpdr_input.repair.repair_controller import waveform_hash,decision_hash
from bpdr_input.reporting.represented_singer_gates import assess
from bpdr_input.modeling.correction_accuracy import assert_independent


def verify_development():
    study.declare();ex=study.ex;root=study.ROOT;folder=root/'arm/development/median11';baseline=study.DEV_BASE
    lineage=ex.read(root/'lineage_integrity_receipt.json');assert lineage['lineage_receipt_sha256']==ex.sha(root/'lineage_receipt.json') and lineage['default_checkpoint_sha256']==ex.sha(study.V8/'final/seed_339/checkpoint.pt')
    lock=ex.read(root/'development_lock.json');assert lock['seed']==339 and lock['lineage_receipt_sha256']==ex.sha(root/'lineage_receipt.json')
    rows=ex.read(study.V8/'development/pairs.json');pairs=study.training.held.prepared(rows);old=ex.read(baseline/'accuracy.json');new=ex.read(folder/'accuracy.json')
    assert len(rows)==len(new['rows'])==len(old['rows'])==48 and sum('excluded' not in p for p in pairs)==24
    neural,payload=study.load(study.V8/'final/seed_339/checkpoint.pt');assert_independent(payload['training_groups'],payload['training_sources'],rows)
    prediction_cache={};replayed_views=0;replayed_arrays=0
    with np.load(baseline/'predictions.npz') as a,np.load(folder/'predictions.npz') as b:
        assert set(a.files)==set(b.files)
        for key in a.files:np.testing.assert_array_equal(a[key],b[key]);replayed_arrays+=1
        for i,p in enumerate(pairs):
            if 'excluded' in p:continue
            for view in ('A','C'):
                path=p['path_'+view]
                if path not in prediction_cache:prediction_cache[path]=np.stack([study.predict_raw(neural,p[view]['views'][0])[0]]*3)
                np.testing.assert_array_equal(prediction_cache[path],b[f'{i}_{view}']);replayed_views+=1
    controls=ex.read(folder/'original_controls.json');original=ex.read(baseline/'original_controls.json');assert len(controls)==len(original)==12
    sources=[ex.read(p)['source'] for p in sorted((study.V8.parent/'2026-10-28_v7_guard/development').glob('source_*.json'))]
    for a,b,source in zip(original,controls,sources,strict=True):
        assert a['source_sha256']==b['source_sha256']==source['source']['sha256'] and a['raw_original_abs_cents']==b['raw_original_abs_cents'] and b['delivered_identity']
        assert a['transaction']['plan']==b['transaction']['plan']
        wave,sr=sf.read(source['working_audio'],dtype='float32');assert sr==22050 and waveform_hash(wave)==b['transaction']['plan']['source_sha256']
        assert decision_hash(b['transaction']['plan'])==b['transaction']['plan']['decision_sha256']
    model=_pitch_model(str((ex.COMPONENT_ROOT/'checkpoints').resolve()));cache={}
    def features(audio,times):
        key=waveform_hash(audio)
        if key not in cache:
            with torch.inference_mode():_,conf,_,act=model(torch.as_tensor(audio,dtype=torch.float32),22050,convert_to_freq=True,return_activations=True)
            a=act.double().reshape(-1,384);center=a.argmax(-1)[:,None];idx=(center+torch.arange(-2,3)).clamp(0,383)
            w=torch.gather(a,1,idx);relative=(w*(idx-center).double()).sum(-1)/w.sum(-1);native=(440*torch.pow(torch.tensor(2.,dtype=torch.float64),(center.squeeze(-1).double()-207+relative)/36)).numpy()
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
                assert decision_hash(trace['permission'])==trace['permission']['decision_sha256']
                plan=trace['permission']
                if not plan['permitted'] or plan['identity']:continue
                audio=p['audio_'+view];times=p[view]['features']['times_seconds'];curve=z[f'{i}_requested_{view}'];reliable=p[view]['reliable']
                candidate,_=render_candidate(audio,22050,times,curve,plan['intervals_seconds'])
                actual=audit(audio,candidate,times,reliable,plan['intervals_seconds'],curve)
                compare(trace['transaction']['post_render_audit'],actual);candidate_count+=1
                if trace['outcome']=='VERIFIED_FIXTURE_CANDIDATE':
                    accepted=trace['transaction']['delivered_wav_audit'];wave,sr=sf.read(files[accepted['sha256']],dtype='float32');assert sr==22050
                    assert waveform_hash(wave)==accepted['decoded_waveform_sha256'] and wave.shape==audio.shape
                    outside=np.ones(len(audio),bool);st=np.arange(len(audio))/22050
                    for lo,hi in plan['intervals_seconds']:outside&=(st<lo)|(st>=hi)
                    assert np.array_equal(wave[outside],audio[outside])
                    realized=audit(audio,wave,times,reliable,plan['intervals_seconds'],curve);compare(accepted['acoustic_audit'],realized);assert realized['passed'];encoded_count+=1
                print('CENTERED_DEVELOPMENT_CASE_VERIFIED',i,view,trace['outcome'],flush=True)
    # Independently recalculate each valid controlled-recovery score, including rollbacks.
    for p,row in zip(pairs,new['rows'],strict=True):
        if 'excluded' in p:continue
        trace=row['trace']['C'];wave=p['audio_C']
        if trace['outcome']=='VERIFIED_FIXTURE_CANDIDATE':wave,_=sf.read(files[trace['transaction']['delivered_wav_audit']['sha256']],dtype='float32')
        mask=p['mask']&(p['target']!=0);before=abs(p['C']['fine'][mask]-p['A']['fine'][mask]).mean();after=abs(fine_pitch(wave,22050,p['C']['features']['times_seconds'])[mask]-p['A']['fine'][mask]).mean()
        np.testing.assert_allclose(before,row['delivered_before_mae_cents'],rtol=0,atol=1e-10);np.testing.assert_allclose(after,row['delivered_after_mae_cents'],rtol=0,atol=1e-10)
        assert row['delivered_verified_improvement']==bool(trace['outcome']=='VERIFIED_FIXTURE_CANDIDATE' and after<before-1e-6)
    gates=assess(new,controls);evidence=ex.read(folder/'evidence.json');assert gates==evidence['gates']
    assert ex.sha(folder/'accuracy.json')==evidence['accuracy_sha256'] and ex.sha(folder/'original_controls.json')==evidence['controls_sha256']
    decision=ex.read(root/'development_decision.json');ref=ex.read(baseline/'evidence.json')
    qualified=bool(new['prediction_gate_passed'] and gates['passed'] and gates['original_controls_all_identity'] and evidence['summary']['delivered_after_mae_cents']<ref['summary']['delivered_after_mae_cents'])
    assert qualified==decision['qualified_on_known_scoreable_scope'] and not decision['fresh_confirmation_access']
    protected=ex.read(root.parent/'2026-10-28_v7_guard/declaration.json')['protected_files']
    for p,h in protected.items():assert ex.sha(p)==h
    assert not (root.parent/'active_model.json').exists();study.declare()
    receipt=dict(attempted=48,scoreable=24,excluded=24,original_controls_identity=12,prediction_views_replayed=replayed_views,prediction_curve_arrays_identical=replayed_arrays,permission_decisions_identical=48,candidate_audits_replayed=candidate_count,encoded_wavs_replayed=encoded_count,valid_reference_scores_recalculated=24,represented_singer_gates_reproduced=True,qualified_on_known_scoreable_scope=qualified,unassessed_groups=gates['unassessed_singers'],fresh_confirmation_audio_accessed=False,protected_files_unchanged=len(protected),evidence_sha256=ex.sha(folder/'evidence.json'),development_decision_sha256=ex.sha(root/'development_decision.json'),verifier_sha256=ex.sha(Path(__file__)),activation_absent=True,research_validation_passed=False)
    ex.write(root/'integrity_receipt.json',receipt);print('CENTERED_DEVELOPMENT_VERIFIED',receipt,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);verify_development()
