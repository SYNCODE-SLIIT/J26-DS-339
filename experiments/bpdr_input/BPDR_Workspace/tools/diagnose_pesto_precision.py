"""Frozen-distribution decoder precision diagnostic; strict audit rules unchanged."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
from importlib.util import find_spec
import numpy as np
import soundfile as sf
import torch
import diagnose_render_disagreement as upstream
from bpdr_input.audio.analysis import _pitch_model
from bpdr_input.modeling.pesto_precision import decode_float64
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.repair_controller import waveform_hash

ROOT=upstream.ROOT.parent/'2026-11-01_pesto_precision'


def declare():
    d=upstream.declare();ex=upstream.study.ex;ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        integrity=ex.read(upstream.ROOT/'integrity_receipt.json');assert integrity['all60_candidates_included']
        source=[Path(__file__).resolve(),Path(decode_float64.__globals__['__file__']).resolve()]
        source+=list(Path(find_spec('pesto').origin).parent.rglob('*.py'))
        results=ex.read(upstream.ROOT/'results.json')
        weight=Path(results['arms']['reference']['cases'][0]['analyzer']['checkpoint_path'])
        hashes={str(p):ex.sha(p) for p in source+[weight,upstream.ROOT/'results.json',upstream.ROOT/'integrity_receipt.json']}
        ex.write(ROOT/'declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_and_input_hashes=hashes,parent_declaration_sha256=ex.sha(upstream.ROOT/'declaration.json'),
            protocol='All60 exact prior candidates,30 per arm. Re-run same frozen float32 PESTO encoder with activations, require exact old native F0/confidence reproduction. Keep argmax/5-bin window/weights; compute weighted pitch and frequency in float64 only. Preserve frame grid,voiced masks,fine track,supports,requested curves and every audit rule including strict per-frame direction. No tolerance, omitted frames, model training or delivered-WAV approval. Verify with separate torch.float64 decoder and report strict-direction disagreement between equivalent implementations.',
            diagnostic_only=True,selection_eligible=False,demo_activated=False,research_validation_passed=False))
    locked=ex.read(ROOT/'declaration.json')
    for p,h in locked['source_and_input_hashes'].items():assert ex.sha(p)==h,p
    assert locked['parent_declaration_sha256']==ex.sha(upstream.ROOT/'declaration.json')
    return locked,d


def track(audio,key,expected_f0,expected_conf):
    ex=upstream.study.ex;out=ROOT/'tracks';out.mkdir(exist_ok=True)
    model=_pitch_model(str((ex.COMPONENT_ROOT/'checkpoints').resolve()))
    assert model.reduction=='alwa' and model.bins_per_semitone==3
    with torch.inference_mode():
        old,confidence,_,activations=model(torch.as_tensor(audio,dtype=torch.float32),22050,convert_to_freq=True,return_activations=True)
    old=old.numpy().reshape(-1);confidence=confidence.numpy().reshape(-1);a=activations.numpy().reshape(-1,384)
    np.testing.assert_array_equal(old,expected_f0);np.testing.assert_array_equal(confidence,expected_conf)
    new=decode_float64(a)
    path=out/(key+'.npz');np.savez_compressed(path,activations=a,old_native=old,new_native=new,confidence=confidence)
    return new,dict(key=key,path=str(path),sha256=ex.sha(path),source_waveform_sha256=waveform_hash(audio),frames=len(old),old_control_exact=True)


def aligned(native,native_times,times):
    idx=np.clip(np.searchsorted(native_times,times),0,len(native_times)-1);previous=np.maximum(idx-1,0)
    idx=np.where(abs(native_times[previous]-times)<abs(native_times[idx]-times),previous,idx)
    return native[idx]


def audit_with(z,before_native,after_native,intervals):
    times=z['times'];features=[]
    for label,native in [('before',before_native),('after',after_native)]:
        voiced=z[label+'_voiced_mask'];f0=aligned(native,z[label+'_native_pitch_times_seconds'],times)
        features.append(dict(times_seconds=times,f0_hz=np.where(voiced,f0,np.nan),voiced_mask=voiced))
    result=audit_curve(*features,z['fine_before'],z['fine_after'],z['reliable'],intervals,z['requested'])
    return json.loads(json.dumps(result,allow_nan=False)),features


def run():
    d,parent=declare();ex=upstream.study.ex
    if (ROOT/'results.json').exists():raise ValueError('Keep completed precision diagnostic')
    rows=ex.read(parent['manifest']);source_cache={};records={};reports={}
    for arm,result in ex.read(upstream.ROOT/'results.json')['arms'].items():
        cases=[]
        for case in result['cases']:
            i=case['case_index'];path=upstream.ROOT/arm/f'case_{i:03d}/tracks.npz'
            assert ex.sha(path)==case['tracks_sha256']
            with np.load(path) as z:
                key=f'source_{i:03d}'
                if key not in source_cache:
                    audio,sr=sf.read(rows[i]['path_C'],dtype='float32');assert sr==22050
                    new,record=track(audio,key,z['before_native_pitch_hz'],z['before_native_pitch_confidence']);source_cache[key]=new
                    record.update(audio_path=rows[i]['path_C'],audio_array=None);records[key]=record
                after_key=f'{arm}_{i:03d}'
                new,record=track(z['candidate_audio'],after_key,z['after_native_pitch_hz'],z['after_native_pitch_confidence'])
                record.update(audio_path=str(path),audio_array='candidate_audio');records[after_key]=record
                audit,_=audit_with(z,source_cache[key],new,case['intervals_seconds'])
            cases.append(dict(case_index=i,before_key=key,after_key=after_key,tracks_sha256=case['tracks_sha256'],old_audit=case['audit'],float64_audit=audit,intervals_seconds=case['intervals_seconds']))
            print('PRECISION_CASE',arm,i,'old/new_wrong',case['audit']['coarse_wrong_direction_frames'],audit['coarse_wrong_direction_frames'],flush=True)
        reports[arm]=dict(cases=cases,old_audit_passes=sum(c['old_audit']['passed'] for c in cases),float64_audit_passes=sum(c['float64_audit']['passed'] for c in cases),old_coarse_wrong_frames=sum(c['old_audit']['coarse_wrong_direction_frames'] for c in cases),float64_coarse_wrong_frames=sum(c['float64_audit']['coarse_wrong_direction_frames'] for c in cases),attempted=200,scoreable=108,excluded=92,rendered_candidates=len(cases))
    declare();ex.write(ROOT/'results.json',dict(declaration_sha256=ex.sha(ROOT/'declaration.json'),tracks=records,arms=reports,diagnostic_only=True,selection_eligible=False,demo_activated=False,research_validation_passed=False))
    print('PRECISION_DIAGNOSTIC_COMPLETE',{k:(v['old_audit_passes'],v['float64_audit_passes']) for k,v in reports.items()},flush=True)


def verify():
    declare();ex=upstream.study.ex;r=ex.read(ROOT/'results.json');alternate={};maxdiff=0.
    model=_pitch_model(str((ex.COMPONENT_ROOT/'checkpoints').resolve()))
    assert model.reduction=='alwa' and model.bins_per_semitone==3
    for key,record in r['tracks'].items():
        if record['audio_array'] is None:audio,sr=sf.read(record['audio_path'],dtype='float32');assert sr==22050
        else:
            with np.load(record['audio_path']) as z:audio=z[record['audio_array']].copy()
        assert waveform_hash(audio)==record['source_waveform_sha256']
        with torch.inference_mode():old,confidence,_,a=model(torch.as_tensor(audio,dtype=torch.float32),22050,convert_to_freq=True,return_activations=True)
        assert ex.sha(record['path'])==record['sha256']
        with np.load(record['path']) as z:
            np.testing.assert_array_equal(a.numpy().reshape(-1,384),z['activations']);np.testing.assert_array_equal(old.numpy().reshape(-1),z['old_native']);np.testing.assert_array_equal(confidence.numpy().reshape(-1),z['confidence'])
            t=a.double().reshape(-1,384);center=torch.argmax(t,dim=-1)[:,None];indices=(center+torch.arange(-2,3)).clamp(0,383)
            weights=torch.gather(t,1,indices);pitch=(weights*(indices.double()/3)).sum(-1)/weights.sum(-1)
            new=(440*torch.pow(torch.tensor(2.,dtype=torch.float64),(pitch-69)/12)).numpy()
            diff=float(np.max(abs(new-z['new_native'])));maxdiff=max(maxdiff,diff)
            np.testing.assert_allclose(new,z['new_native'],rtol=0,atol=2e-10)
            alternate[key]=new
    mismatches=[];cases=0
    for arm,result in r['arms'].items():
        for c in result['cases']:
            p=upstream.ROOT/arm/f"case_{c['case_index']:03d}/tracks.npz"
            assert ex.sha(p)==c['tracks_sha256']
            with np.load(p) as z:
                with np.load(r['tracks'][c['before_key']]['path']) as b,np.load(r['tracks'][c['after_key']]['path']) as a:
                    same,_=audit_with(z,b['new_native'],a['new_native'],c['intervals_seconds']);assert same==c['float64_audit']
                audit,features=audit_with(z,alternate[c['before_key']],alternate[c['after_key']],c['intervals_seconds'])
                # Strict signs must agree frame by frame, not merely in aggregate count.
                before,after=features
                _,opposite,_,_=upstream.direction_details(before,after,z['fine_before'],z['fine_after'],z['reliable'],c['intervals_seconds'],z['requested'])
                with np.load(r['tracks'][c['before_key']]['path']) as b,np.load(r['tracks'][c['after_key']]['path']) as a:
                    _,f= audit_with(z,b['new_native'],a['new_native'],c['intervals_seconds'])
                    _,expected,_,_=upstream.direction_details(*f,z['fine_before'],z['fine_after'],z['reliable'],c['intervals_seconds'],z['requested'])
                if not np.array_equal(opposite,expected) or audit['reasons']!=same['reasons']:mismatches.append(dict(arm=arm,case_index=c['case_index']))
                cases+=1
    protected=ex.read(ROOT.parent/'2026-10-28_v7_guard/declaration.json')['protected_files']
    for p,h in protected.items():assert ex.sha(p)==h
    declare();receipt=dict(tracks_reinferred=len(alternate),candidate_audits_replayed=cases,max_numpy_torch_decoder_difference_hz=maxdiff,strict_sign_implementation_disagreements=mismatches,strict_sign_implementation_agreement=not mismatches,arithmetic_check_bound_is_not_acceptance_tolerance=True,protected_files_unchanged=len(protected),results_sha256=ex.sha(ROOT/'results.json'),diagnostic_only=True,selection_eligible=False,model_activated=False,research_validation_passed=False)
    ex.write(ROOT/'integrity_receipt.json',receipt);print('PRECISION_DIAGNOSTIC_VERIFIED',receipt,flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['run','verify']);args=p.parse_args()
    run() if args.stage=='run' else verify()
