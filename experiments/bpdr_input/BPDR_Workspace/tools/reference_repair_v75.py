"""V75 stage 2 of the reference-conditioned review design: repair-to-reference through the EXISTING delivery path.
No fit. Curve source differs from V66/V69 only here: predictions = -(pitch_injected - pitch_reference) on frames reliable in both
(0 elsewhere), fed as 3 identical views to short_phrase_policy.proposal (same activation/protection/support rules, same renderer,
same three-tracker audit, same encoded-WAV audit and rollback). Reference = unshifted original's analysis pitch (exact by construction).
Scoring identical to V47/V66: verified beneficial needs audit pass, lower plateau error, known-zero not worse by >1e-6c; harm otherwise.
Rule (declared before scores): passes iff singer-macro after-MAE<=8c AND verified_beneficial/scoreable>=0.70 AND known_zero_harms/scoreable<=0.10
AND every original control waveform unchanged. Not natural-intent evidence; reference exactness is by construction."""
from dataclasses import replace
from collections import Counter
import sys,json
import numpy as np
import soundfile as sf
import compare_tail_preservation as run
import evaluate_paired_short_delivery as ev
from bpdr_input.repair import short_phrase_policy as policy,experimental_short_audit as audit_module
from bpdr_input.repair.framewise_renderer import render_candidate
from bpdr_input.repair.centered_precision_audit import coarse_features
from bpdr_input.repair.rule_correction import fine_hz,fine_pitch
from bpdr_input.repair.repair_controller import transact_fixture,waveform_hash
from bpdr_input.repair.learned_runtime import audit_delivered_wav
from evaluate_short_phrase_delivery import load_input
OUT=run.WS/'results/transformer_usability/2026-11-04_v75_reference_repair'
def main():
    if sys.argv[1:]==['declare']:
        run.write(OUT/'declaration.json',dict(doc=__doc__,tool_sha256=run.sha(__file__)));print('DECLARED');return
    assert run.read(OUT/'declaration.json')['tool_sha256']==run.sha(__file__)
    d,rows,originals=run.declare();data=run.paired.records(rows,originals);by_id={s['source_id']:s for s in originals if 'excluded' not in s}
    cache={};fcache={}
    def measured(a,t):
        k=waveform_hash(a)
        if k not in cache:cache[k]=(coarse_features(a,22050,run.WS/'checkpoints'),fine_hz(a,22050,t),audit_module.cross_correlation_pitch(a,22050,t))
        return cache[k]
    def fine(a,t):
        k=waveform_hash(a)
        if k not in fcache:fcache[k]=fine_pitch(a,22050,t)
        return fcache[k]
    reports=[]
    for rec in data:
        folder=OUT/rec['id'];folder.mkdir(parents=True,exist_ok=True)
        audio,times,views,reliable=load_input(rec,rows,originals)
        raw=rec['raw'];pred=np.zeros(len(raw),np.float32)
        if rec['kind']=='corrupted':
            with np.load(by_id[rec['source_id']]['analysis_A']) as z:ref=z['raw_views'][0]
            assert len(ref)==len(raw);both=(ref[:,66]>.5)&(raw[:,66]>.5);pred[both]=-(raw[both,64]-ref[both,64])
        proposal=policy.proposal(audio,times,views,np.stack([pred]*3));curve=proposal['curve'];plan=proposal['plan']
        def auditor(b,a,sr,iv):
            bf,yb,cb=measured(b,times);af,ya,ca=measured(a,times);return audit_module.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,iv,curve)
        renderer=lambda a,sr,iv:render_candidate(a,sr,times,curve,iv,ev.RENDER)[0]
        out,tr=transact_fixture(audio,22050,plan,renderer,auditor);verified=None
        if tr['outcome']=='VERIFIED_FIXTURE_CANDIDATE':
            p=folder/'candidate.wav';sf.write(p,out,22050,subtype='PCM_24')
            try:tr['encoded_audit']=audit_delivered_wav(p,audio,22050,plan['intervals_seconds'],auditor);out,_=sf.read(p,dtype='float32');verified=str(p)
            except Exception as e:tr.update(outcome='ROLLED_BACK_ENCODED',rollback=True,encoded_error=str(e));out=audio.copy()
        r=dict(id=rec['id'],group=rec['group'],kind=rec['kind'],outcome=tr['outcome'],verified=bool(verified),unchanged=bool(np.array_equal(audio,out)),
            changed_samples_outside_plan=None)
        if rec['kind']=='corrupted':
            row=rows[rec['pair_index']];reference,_=sf.read(by_id[row['source_id']]['path_A'],dtype='float32')
            with np.load(row['target_path']) as z:pos=z['plateau'];zero=z['known_zero']
            b=fine(audio,times)-fine(reference,times);a=fine(out,times)-fine(reference,times)
            bm,am=float(np.mean(abs(b[pos]))),float(np.mean(abs(a[pos])));bz,az=float(np.mean(abs(b[zero]))),float(np.mean(abs(a[zero])))
            gain=bool(verified and am<bm-1e-6);col=bool(verified and az>bz+1e-6)
            r.update(before=bm,after=am,bz=bz,az=az,beneficial=bool(gain and not col),zero_harm=col,harm=bool(verified and (am>bm+1e-6 or col)))
        run.write(folder/'result.json',r);reports.append(r)
    sc=[r for r in reports if r['kind']=='corrupted'];og=[r for r in reports if r['kind']=='original']
    groups=sorted({r['group'] for r in sc});mac=lambda f:float(np.mean([np.mean([f(r) for r in sc if r['group']==g]) for g in groups]))
    res=dict(scoreable=len(sc),macro_before=mac(lambda r:r['before']),macro_after=mac(lambda r:r['after']),beneficial=sum(r['beneficial'] for r in sc),
        harms=sum(r['harm'] for r in sc),zero_harms=sum(r['zero_harm'] for r in sc),verified=sum(r['verified'] for r in sc),outcomes=dict(Counter(r['outcome'] for r in reports)),
        originals=len(og),originals_unchanged=all(r['unchanged'] for r in og),
        by_group={g:dict(n=sum(r['group']==g for r in sc),beneficial=sum(r['beneficial'] for r in sc if r['group']==g),zero_harms=sum(r['zero_harm'] for r in sc if r['group']==g),
            after=float(np.mean([r['after'] for r in sc if r['group']==g]))) for g in groups})
    res['passes']=bool(res['macro_after']<=8 and res['beneficial']/res['scoreable']>=.7 and res['zero_harms']/res['scoreable']<=.1 and res['originals_unchanged'])
    run.write(OUT/'results.json',dict(res,records=reports,declaration_sha256=run.sha(OUT/'declaration.json'),no_fit=True));print('V75',json.dumps({k:v for k,v in res.items() if k!='by_group'}),flush=True)
main()
