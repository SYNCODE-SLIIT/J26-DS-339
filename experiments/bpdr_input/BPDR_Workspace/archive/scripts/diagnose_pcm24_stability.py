"""All-candidate known-development PCM24 stability; diagnostic only."""
import argparse,io,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import soundfile as sf
import torch
import qualify_precision_development as parent
import diagnose_render_disagreement as detail
from bpdr_input.audio.analysis import _pitch_model
from bpdr_input.repair.precision_audit import coarse_features
from bpdr_input.repair.framewise_renderer import render_candidate
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.rule_correction import fine_hz
from bpdr_input.repair.repair_controller import waveform_hash,decision_hash

ROOT=parent.ROOT.parent/'2026-11-01_pcm24_stability'
FOLDER=parent.ROOT/'arm/development/median11'


def declare():
    parent.declare();ex=parent.ex;ROOT.mkdir(exist_ok=True)
    if not (ROOT/'declaration.json').exists():
        receipt=ex.read(parent.ROOT/'integrity_receipt.json');assert receipt['candidate_audits_replayed']==11
        paths=[Path(__file__).resolve(),parent.ROOT/'integrity_receipt.json',parent.ROOT/'qualification_status.json',FOLDER/'accuracy.json',FOLDER/'predictions.npz']
        paths+=list((FOLDER/'audio').rglob('*_candidate.wav'))
        ex.write(ROOT/'declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_and_input_hashes={str(p):ex.sha(p) for p in paths},parent_declaration_sha256=ex.sha(parent.ROOT/'declaration.json'),
            protocol='All11 permitted known-development C candidates, including accepted controls. Reconstruct fixed Praat float32 candidates, encode PCM24 in private memory, compare saved decoded bytes when existing. Measure source/float/PCM24 using unchanged precision auditor and record full strict-sign masks/timing/magnitudes,quantization and native voicing/confidence. Pre-withheld candidates receive hypothetical encoding probes only, never an accepted output. Separate torch.float64 inference/decoding verification; no model/gate/threshold/support/denominator/selection changes.',attempted=48,scoreable=24,excluded=24,diagnostic_only=True,selection_eligible=False,demo_activated=False,research_validation_passed=False))
    d=ex.read(ROOT/'declaration.json')
    for p,h in d['source_and_input_hashes'].items():assert ex.sha(p)==h,p
    assert d['parent_declaration_sha256']==ex.sha(parent.ROOT/'declaration.json')
    return d


def torch_features(audio,times):
    model=_pitch_model(str((parent.ex.COMPONENT_ROOT/'checkpoints').resolve()))
    with torch.inference_mode():_,confidence,_,act=model(torch.as_tensor(audio,dtype=torch.float32),22050,convert_to_freq=True,return_activations=True)
    a=act.double().reshape(-1,384);center=a.argmax(-1)[:,None];index=(center+torch.arange(-2,3)).clamp(0,383);weight=torch.gather(a,1,index)
    pitch=(weight*(index.double()/3)).sum(-1)/weight.sum(-1);native=(440*torch.pow(torch.tensor(2.,dtype=torch.float64),(pitch-69)/12)).numpy()
    conf=confidence.numpy().reshape(-1);nt=np.arange(len(native))*(round(model.hop_size*22050/1000)/22050)
    i=np.clip(np.searchsorted(nt,times),0,len(nt)-1);previous=np.maximum(i-1,0);i=np.where(abs(nt[previous]-times)<abs(nt[i]-times),previous,i)
    voiced=(conf[i]>=.8)&(native[i]>0)
    return dict(times_seconds=times,f0_hz=np.where(voiced,native[i],np.nan),confidence=conf[i],voiced_mask=voiced,native_pitch_hz=native,native_pitch_times_seconds=nt,native_pitch_confidence=conf)


def measured(audio,times,independent=False):
    features=torch_features(audio,times) if independent else coarse_features(audio,22050,parent.ex.COMPONENT_ROOT/'checkpoints')
    assert np.array_equal(features['times_seconds'],times)
    return features,fine_hz(audio,22050,times)


def audit_and_details(before,after,reliable,intervals,requested):
    bf,yb=before;af,ya=after
    a=json.loads(json.dumps(audit_curve(bf,af,yb,ya,reliable,intervals,requested),allow_nan=False))
    checked,opposing,fine,coarse=detail.direction_details(bf,af,yb,ya,reliable,intervals,requested)
    frames=[dict(frame=int(j),time_seconds=float(bf['times_seconds'][j]),requested_cents=float(requested[j]),fine_shift_cents=float(fine[j]),coarse_shift_cents=float(coarse[j]),coarse_before_hz=float(bf['f0_hz'][j]),coarse_after_hz=float(af['f0_hz'][j]),confidence_before=float(bf['confidence'][j]),confidence_after=float(af['confidence'][j])) for j in np.flatnonzero(opposing)]
    return a,frames,checked,opposing


def run():
    declare();ex=parent.ex
    if (ROOT/'results.json').exists():raise ValueError('Retain completed stability diagnosis')
    pairs=parent.training.held.prepared(ex.read(parent.V8/'development/pairs.json'));result=ex.read(FOLDER/'accuracy.json');reports=[]
    with np.load(FOLDER/'predictions.npz') as z:
        for i,(p,row) in enumerate(zip(pairs,result['rows'],strict=True)):
            if 'excluded' in p:continue
            trace=row['trace']['C'];plan=trace['permission']
            if not plan['permitted'] or plan['identity']:continue
            assert decision_hash(plan)==plan['decision_sha256']
            out=ROOT/f'case_{i:02d}';out.mkdir(exist_ok=True)
            audio=p['audio_C'];times=p['C']['features']['times_seconds'];curve=z[f'{i}_requested_C'];intervals=plan['intervals_seconds'];reliable=p['C']['reliable']
            candidate,_=render_candidate(audio,22050,times,curve,intervals)
            stream=io.BytesIO();sf.write(stream,candidate,22050,format='WAV',subtype='PCM_24');encoded=stream.getvalue();decoded,sr=sf.read(io.BytesIO(encoded),dtype='float32');assert sr==22050
            existing=FOLDER/f'audio/case_{i:02d}/C_candidate.wav'
            if existing.exists():
                old,sr=sf.read(existing,dtype='float32');assert sr==22050;np.testing.assert_array_equal(decoded,old)
            observations={name:measured(w,times) for name,w in [('source',audio),('float',candidate),('pcm24',decoded)]}
            audits={};frame_rows={};arrays=dict(source_audio=audio,float_audio=candidate,pcm24_audio=decoded,times=times,requested=curve,reliable=reliable)
            for name in ('float','pcm24'):
                a,f,checked,opposing=audit_and_details(observations['source'],observations[name],reliable,intervals,curve);audits[name]=a;frame_rows[name]=f;arrays[name+'_checked']=checked;arrays[name+'_opposing']=opposing
            original=json.loads(json.dumps(trace['transaction']['post_render_audit']));original.pop('coarse_measurement',None)
            assert audits['float']==original
            if trace['outcome']=='VERIFIED_FIXTURE_CANDIDATE':assert audits['pcm24']['passed']
            if trace['transaction'].get('delivered_error')=='ENCODED_CANDIDATE_ACOUSTIC_AUDIT_FAILED':assert not audits['pcm24']['passed']
            for name,(features,fine) in observations.items():
                arrays[name+'_fine_hz']=fine
                for k in ('f0_hz','confidence','voiced_mask','native_pitch_hz','native_pitch_times_seconds','native_pitch_confidence'):arrays[name+'_'+k]=features[k]
            np.savez_compressed(out/'tracks.npz',**arrays)
            st=np.arange(len(audio))/22050;inside=np.zeros(len(audio),bool)
            for lo,hi in intervals:inside|=(st>=lo)&(st<hi)
            assert np.array_equal(audio[~inside],candidate[~inside]) and np.array_equal(audio[~inside],decoded[~inside])
            r=dict(case_index=i,group=row['group'],existing_outcome=trace['outcome'],intervals_seconds=intervals,original_render_candidate_sha256=waveform_hash(candidate),saved_encoded_candidate_exists=existing.exists(),encoded_probe_was_actual_pipeline_output=existing.exists(),float_audit=audits['float'],pcm24_audit=audits['pcm24'],opposing_frames=frame_rows,max_sample_quantization_error=float(np.max(abs(candidate-decoded))),samples_changed_by_quantization=int(np.count_nonzero(candidate!=decoded)),outside_samples_unchanged=True,tracks_sha256=ex.sha(out/'tracks.npz'),diagnostic_only=True,accepted_for_delivery=False)
            ex.write(out/'receipt.json',r);reports.append(r)
            print('PCM24_CASE',i,'float/encoded',audits['float']['reasons'],audits['pcm24']['reasons'],flush=True)
    assert len(reports)==11
    declare();ex.write(ROOT/'results.json',dict(cases=reports,attempted=48,scoreable=24,excluded=24,float_passes=sum(r['float_audit']['passed'] for r in reports),pcm24_probe_passes=sum(r['pcm24_audit']['passed'] for r in reports),declaration_sha256=ex.sha(ROOT/'declaration.json'),diagnostic_only=True,selection_eligible=False,model_activated=False,research_validation_passed=False))
    print('PCM24_DIAGNOSIS_COMPLETE',flush=True)


def verify():
    declare();ex=parent.ex;r=ex.read(ROOT/'results.json');assert r['declaration_sha256']==ex.sha(ROOT/'declaration.json');cases=0;variants=0;strict_disagreements=[]
    pairs=parent.training.held.prepared(ex.read(parent.V8/'development/pairs.json'))
    for row in r['cases']:
        i=row['case_index'];path=ROOT/f'case_{i:02d}/tracks.npz';assert ex.sha(path)==row['tracks_sha256'];p=pairs[i]
        with np.load(path) as z:
            np.testing.assert_array_equal(p['audio_C'],z['source_audio']);np.testing.assert_array_equal(p['C']['reliable'],z['reliable']);times=z['times'];curve=z['requested'];intervals=row['intervals_seconds']
            candidate,_=render_candidate(p['audio_C'],22050,times,curve,intervals);np.testing.assert_array_equal(candidate,z['float_audio'])
            stream=io.BytesIO();sf.write(stream,candidate,22050,format='WAV',subtype='PCM_24');encoded,_=sf.read(io.BytesIO(stream.getvalue()),dtype='float32');np.testing.assert_array_equal(encoded,z['pcm24_audio'])
            independent={name:measured(z[name+'_audio'],times,True) for name in ('source','float','pcm24')}
            for name,(features,fine) in independent.items():
                np.testing.assert_array_equal(fine,z[name+'_fine_hz'])
                for k in ('f0_hz','native_pitch_hz'):np.testing.assert_allclose(features[k],z[name+'_'+k],rtol=0,atol=2e-10,equal_nan=True)
                for k in ('confidence','voiced_mask','native_pitch_times_seconds','native_pitch_confidence'):np.testing.assert_array_equal(features[k],z[name+'_'+k])
                variants+=1
            for name in ('float','pcm24'):
                audit,frames,checked,opposing=audit_and_details(independent['source'],independent[name],z['reliable'],intervals,curve)
                expected=row[name+'_audit'];same=np.array_equal(checked,z[name+'_checked']) and np.array_equal(opposing,z[name+'_opposing']) and audit['reasons']==expected['reasons']
                if not same:strict_disagreements.append(dict(case_index=i,variant=name))
                for k in ('fine','coarse'):
                    for metric in ('median_error_cents','p90_error_cents'):np.testing.assert_allclose(audit[k][metric],expected[k][metric],rtol=0,atol=1e-8)
            assert row['outside_samples_unchanged'];cases+=1
        print('PCM24_CASE_REPLAYED',i,flush=True)
    assert cases==11 and variants==33
    protected=ex.read(ROOT.parent/'2026-10-28_v7_guard/declaration.json')['protected_files']
    for p,h in protected.items():assert ex.sha(p)==h
    assert not (ROOT.parent/'active_model.json').exists();declare()
    receipt=dict(candidates_replayed=cases,source_float_pcm24_measurements_reinferred=variants,strict_sign_backend_disagreements=strict_disagreements,strict_sign_backend_agreement=not strict_disagreements,protected_files_unchanged=len(protected),results_sha256=ex.sha(ROOT/'results.json'),verifier_source_sha256=ex.sha(Path(__file__)),diagnostic_only=True,selection_eligible=False,activation_absent=True,research_validation_passed=False)
    ex.write(ROOT/'integrity_receipt.json',receipt);print('PCM24_DIAGNOSIS_VERIFIED',receipt,flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['run','verify']);args=parser.parse_args()
    run() if args.stage=='run' else verify()
