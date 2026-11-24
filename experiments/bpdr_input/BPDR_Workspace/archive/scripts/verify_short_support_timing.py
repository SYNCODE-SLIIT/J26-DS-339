"""Replay every timing-probe decoded audit, exclusion and opposite-request control."""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import json,sys
import numpy as np
import soundfile as sf
import torch

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
from acquire_fresh_confirmation import check_lock
from probe_short_phrase_renderer import sha,phrase
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.repair.rule_correction import fine_hz,segment_notes,RulePolicy
from bpdr_input.repair.centered_precision_audit import coarse_features
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.interventions import AuditPolicy
from bpdr_input.repair.framewise_renderer import render_candidate,RendererPolicy

ROOT=WS/'results/transformer_usability/2026-11-02_v39_short_timing'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def norm(x):return json.loads(json.dumps(x,allow_nan=False))


def main():
    torch.set_num_threads(2);check_lock()
    d=read(ROOT/'declaration.json');r=read(ROOT/'results.json')
    assert r['declaration_sha256']==sha(ROOT/'declaration.json')
    for path,h in d['source_hashes'].items():assert sha(path)==h,path
    assert len(r['cases'])==384 and len(r['originals'])==28
    measured={};source_data={};exclusions=0;replayed=0;opposite_rejections=0
    for source in r['originals']:
        folder=ROOT/source['id'];path=folder/'original.wav'
        audio,sr=sf.read(path,dtype='float32');assert sr==22050 and sha(path)==source['working_sha256']
        if source['kind']=='real_voice':
            index=int(source['id'].split('_')[-1])-1;record=d['source_records'][index]
            assert sha(record['working_audio'])==record['working_sha256']
            original,rate=sf.read(record['working_audio'],dtype='float32')
            assert rate==sr and np.array_equal(audio,original[:10*sr])
            m=recovery.measured(audio)
            regions=segment_notes(m['views'][0][:,64],m['reliable'],RulePolicy(note_split_cents=70.))
            region=max(regions,key=lambda p:p[1]-p[0]) if regions else None
            assert norm(region)==source['longest_region_frames']
        else:
            _,base,width,phase=source['id'].split('_')
            fixture,_=phrase(int(base),float(width));region=None
            # Quantization error is bounded independently; decoded source is the audit identity.
            assert np.max(abs(audio-fixture))<=2**-23
        times=np.arange(len(audio)//441+1)*.02
        identity,_=render_candidate(audio,sr,times,np.zeros(len(times)),[],RendererPolicy())
        assert np.array_equal(identity,audio)
        source_data[source['id']]=(audio,times,region)
    def measure(audio,times):
        key=__import__('hashlib').sha256(audio.tobytes()).hexdigest()
        if key not in measured:measured[key]=(coarse_features(audio,22050,WS/'checkpoints'),fine_hz(audio,22050,times))
        return measured[key]
    for row in r['cases']:
        audio,times,region=source_data[row['id']];width=row['width_seconds'];cents=row['cents']
        if row['kind']=='real_voice':
            valid=region is not None and (region[1]-region[0])*.02>=width+.14-1e-12
            center=float(np.mean(region)*.02) if region else None
        else:
            _,base,w,phase=row['id'].split('_');_,support=phrase(int(base),float(w))
            center=float(np.mean(support)+float(phase));valid=True
        expected=(float(center-width/2),float(center+width/2)) if center is not None else None
        assert row['support']==norm(expected)
        if not valid:
            assert row['excluded']=='NO_SINGLE_RELIABLE_REGION_WITH_70MS_CONTEXT'
            exclusions+=1;continue
        assert 'excluded' not in row,row
        path=Path(row['output']);assert sha(path)==row['output_sha256']
        decoded,sr=sf.read(path,dtype='float32');assert sr==22050 and audio.shape==decoded.shape
        assert np.isfinite(decoded).all() and abs(decoded).max()<1
        t=np.arange(len(audio))/sr;outside=(t<=expected[0])|(t>=expected[1])
        assert np.array_equal(audio[outside],decoded[outside])
        assert np.any(audio[~outside]!=decoded[~outside])
        before,yb=measure(audio,times);after,ya=measure(decoded,times)
        reliable=before['voiced_mask']&np.isfinite(yb)&(yb>0)
        curve=np.where((times>=expected[0])&(times<expected[1]),cents,0.)
        policy=AuditPolicy(**row['audit_policy'])
        # Every non-timing numerical gate remains exactly the historical value.
        old=norm(asdict(AuditPolicy()));current=norm(asdict(policy))
        for key,value in old.items():
            if key not in ['version','ramp_seconds','analysis_guard_seconds']:assert current[key]==value,key
        audit=audit_curve(before,after,yb,ya,reliable,[expected],curve,policy)
        opposite=audit_curve(before,after,yb,ya,reliable,[expected],-curve,policy)
        assert norm(audit)==row['audit'] and norm(opposite)==row['opposite_request_audit']
        assert not opposite['passed'];opposite_rejections+=1;replayed+=1
    counts=Counter()
    for row in r['cases']:
        counts[(row['kind'],row['arm'],'attempted')]+=1
        if 'audit' in row:counts[(row['kind'],row['arm'],'rendered')]+=1
        if row.get('audit',{}).get('passed'):counts[(row['kind'],row['arm'],'passed')]+=1
    receipt=dict(originals_recreated=28,audits_replayed=replayed,construction_exclusions_replayed=exclusions,
        opposite_request_rejections=opposite_rejections,outside_samples_exact=True,
        non_timing_audit_gates_unchanged=True,v37_runtime_closure_unchanged=True,
        verifier_source_sha256=sha(__file__),results_sha256=sha(ROOT/'results.json'),
        model_accuracy_assessed=False,production_eligible=False,
        counts=[dict(kind=k[0],arm=k[1],metric=k[2],count=v) for k,v in sorted(counts.items())])
    (ROOT/'integrity_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('SHORT_TIMING_REPLAY_PASS',replayed,exclusions,opposite_rejections,flush=True)


if __name__=='__main__':main()
