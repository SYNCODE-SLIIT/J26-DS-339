"""V78 diagnostic only (no scores, no rule): rerun V77 settings on the 45 ROLLED_BACK cases plus all S10/S7 corrupted cases and record audit/transaction reasons.
Earlier text: V77 (one change vs V76, reference-conditioned mode only): guard 4->2 frames, min support .24->.20 s in policy, controller and renderer. Chosen after the V76 breakdown (86 corrupted IDENTITY: PROTECTED_REGION 55, INSUFFICIENT_SUPPORT 28), so results are ADAPTIVE to this pool and optimistic, not fresh evidence. Blind-model policy unchanged.
Rule declared before scores: helps iff verified beneficial > 294 AND count of harms with known-zero increase >0.1c <= 5% of scoreable AND originals unchanged. Strict 1e-6c harm count and V75 pass criteria also reported.
Earlier text: V76 (one change vs V75: applied correction clipped to +-30c before the policy, so tracker noise on a 30c deviation no longer trips CORRECTION_LIMIT; limit itself unchanged).
Rule declared before scores (report recommendation 2 stopping rule): helps iff verified beneficial > 174 (V75) AND known-zero harms/scoreable <= 0.054 AND all originals unchanged. Also report V75 pass criteria (macro<=8c, beneficial>=70%, zero harm<=10%). No sweep of the cap.
V75 text follows. V75 stage 2 of the reference-conditioned review design: repair-to-reference through the EXISTING delivery path.
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
import inspect
from dataclasses import replace as _r
SRC=inspect.getsource(policy.proposal).replace("if policy!=ShortPolicy():raise ValueError('Unregistered diagnostic short policy')",'').replace('min_support_seconds=.24','min_support_seconds=policy.min_support_seconds')
_ns=dict(policy.__dict__);exec(SRC,_ns);proposal2=_ns['proposal']
POL=policy.ShortPolicy(protected_guard_frames=2,min_support_seconds=.20)
ev.RENDER=_r(ev.RENDER,min_support_seconds=.20)
OUT=run.WS/'results/transformer_usability/2026-11-05_v78_audit_reasons'
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
    V77=run.read(run.WS/'results/transformer_usability/2026-11-05_v77_reference_guard/results.json')['records']
    SEL={r['id'] for r in V77 if r['kind']=='corrupted' and (r['outcome']=='ROLLED_BACK' or r['group'] in('S10','S7'))}
    reasons={}
    for rec in data:
        if rec['id'] not in SEL:continue
        folder=OUT/rec['id'];folder.mkdir(parents=True,exist_ok=True)
        audio,times,views,reliable=load_input(rec,rows,originals)
        raw=rec['raw'];pred=np.zeros(len(raw),np.float32)
        if rec['kind']=='corrupted':
            with np.load(by_id[rec['source_id']]['analysis_A']) as z:ref=z['raw_views'][0]
            assert len(ref)==len(raw);both=(ref[:,66]>.5)&(raw[:,66]>.5);pred[both]=np.clip(-(raw[both,64]-ref[both,64]),-30.,30.)
        proposal=proposal2(audio,times,views,np.stack([pred]*3),POL);curve=proposal['curve'];plan=proposal['plan']
        def auditor(b,a,sr,iv):
            bf,yb,cb=measured(b,times);af,ya,ca=measured(a,times);return audit_module.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,iv,curve)
        renderer=lambda a,sr,iv:render_candidate(a,sr,times,curve,iv,ev.RENDER)[0]
        out,tr=transact_fixture(audio,22050,plan,renderer,auditor);verified=None
        if tr['outcome']=='VERIFIED_FIXTURE_CANDIDATE':
            p=folder/'candidate.wav';sf.write(p,out,22050,subtype='PCM_24')
            try:tr['encoded_audit']=audit_delivered_wav(p,audio,22050,plan['intervals_seconds'],auditor);out,_=sf.read(p,dtype='float32');verified=str(p)
            except Exception as e:tr.update(outcome='ROLLED_BACK_ENCODED',rollback=True,encoded_error=str(e));out=audio.copy()
        reasons[rec['id']]=dict(group=rec['group'],outcome=tr['outcome'],plan_reasons=plan.get('reasons'),cand=[c['reason'] for c in proposal['candidates']],tr={k:v for k,v in tr.items() if k not in('plan','accepted_audio','diagnostics')})
        r=dict(id=rec['id'],group=rec['group'],kind=rec['kind'],outcome=tr['outcome'],verified=bool(verified),unchanged=bool(np.array_equal(audio,out)),
            changed_samples_outside_plan=None)
        if rec['kind']=='corrupted':
            row=rows[rec['pair_index']];reference,_=sf.read(by_id[row['source_id']]['path_A'],dtype='float32')
            with np.load(row['target_path']) as z:pos=z['plateau'];zero=z['known_zero']
            b=fine(audio,times)-fine(reference,times);a=fine(out,times)-fine(reference,times)
            bm,am=float(np.mean(abs(b[pos]))),float(np.mean(abs(a[pos])));bz,az=float(np.mean(abs(b[zero]))),float(np.mean(abs(a[zero])))
            gain=bool(verified and am<bm-1e-6);col=bool(verified and az>bz+1e-6)
            r.update(before=bm,after=am,bz=bz,az=az,beneficial=bool(gain and not col),zero_harm=col,harm01=bool(verified and az>bz+.1),harm=bool(verified and (am>bm+1e-6 or col)))
        run.write(folder/'result.json',r);reports.append(r)
    run.write(OUT/'reasons.json',reasons);print('V78',len(reasons),flush=True)
main()
