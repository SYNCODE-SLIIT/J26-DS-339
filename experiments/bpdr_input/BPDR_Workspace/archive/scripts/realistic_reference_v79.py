"""V79: realistic-reference test. No fit, no rendering, existing stored features only.
STAND-IN reference (not a BPDR inference of notes; stands for what an upstream notes/key owner might supply): from the UNALTERED original, segment reliable
frames into notes (segment_notes, note_split_cents=70), note median pitch, per-recording tuning offset = median of the wrapped (median mod 100c) residuals,
reference = offset + 100*round((median-offset)/100) (nearest semitone on the recording's own grid).
(a) Natural false-flag rate on the unaltered originals: a note region of >=12 frames is FLAGGED if it contains >=12 consecutive reliable frames with |pitch-ref|>=12c of one sign.
    Reported: share of regions flagged (singer-macro over groups with originals) and share of originals with any flag.
(b) Pairs (445): deviation of the injected clip against the ORIGINAL's reference. Detected = a sustained run (>=min(width,12) frames... =12 frames when width>=12) of |dev|>=12c overlapping the injected span.
    Clean = detected AND no sustained flagged run elsewhere (outside span +-5 frames). Singer-macro.
(c) Per-group paired AUROC: injected-span median |dev| vs median |dev| of same-width spans inside natural regions of the same group's originals.
Rule (declared before scores): silent reference repair viable iff natural region flag rate(macro)<=0.10 AND pair detection(macro)>=0.80 AND clean(macro)>=0.60.
Otherwise: review-only (user accepts each flag). Reference exactness is NOT assumed; this is also not natural-intent accuracy."""
import sys,json
import numpy as np
import compare_tail_preservation as run
from bpdr_input.repair.rule_correction import RulePolicy
from bpdr_input.repair.held_voice_policy import spans as _spans
OUT=run.WS/'results/transformer_usability/2026-11-05_v79_realistic_reference'
ACT=12.;MINF=12
def regions(p,rel):
    seg=run.paired.segment_notes
    return [(a,b) for a,b in seg(p,rel,RulePolicy(note_split_cents=70.)) if b-a>=1]
def reference(p,rel):
    R=regions(p,rel);meds=[float(np.median(p[a:b])) for a,b in R]
    if not R:return None,[]
    off=float(np.median([((m+50)%100)-50 for m in meds]));ref=np.full(len(p),np.nan)
    for (a,b),m in zip(R,meds):ref[a:b]=off+100*np.round((m-off)/100)
    return ref,R
def sustained(dev,ok,a,b):
    """runs >=MINF of one sign with |dev|>=ACT inside [a,b); returns list of (s,e)"""
    out=[]
    for sign in(-1,1):
        on=(dev*sign>=ACT)&ok;i=a
        while i<b:
            if on[i]:
                j=i
                while j<b and on[j]:j+=1
                if j-i>=MINF:out.append((i,j))
                i=j
            else:i+=1
    return out
def auroc(a,b):
    a=np.asarray(a);b=np.asarray(b);return float(np.mean((a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])))
def main():
    if sys.argv[1:]==['declare']:
        run.write(OUT/'declaration.json',dict(doc=__doc__,tool_sha256=run.sha(__file__)));print('DECLARED');return
    assert run.read(OUT/'declaration.json')['tool_sha256']==run.sha(__file__)
    d,rows,originals=run.declare();data=run.paired.records(rows,originals)
    O={};G={}
    for r in data:
        if r['kind']!='original':continue
        p=r['raw'][:,64];rel=r['raw'][:,66]>.5;ref,R=reference(p,rel)
        if ref is None:continue
        dev=p-ref;ok=rel&np.isfinite(dev);O[r['source_id']]=(ref,R,dev,ok)
        g=G.setdefault(r['group'],dict(reg=[],orig=[],nat={}))
        for a,b in R:
            if b-a>=MINF:g['reg'].append(bool(sustained(dev,ok,a,b)))
            for w in(12,14,18,25):
                for s in range(a,b-w+1,3):
                    if ok[s:s+w].all():g['nat'].setdefault(w,[]).append(float(np.median(abs(dev[s:s+w]))))
        g['orig'].append(any(sustained(dev,ok,a,b) for a,b in R))
    P={}
    for r in data:
        if r['kind']!='corrupted' or r['source_id'] not in O:continue
        ref,R,_,_=O[r['source_id']];p=r['raw'][:,64];rel=r['raw'][:,66]>.5
        dev=p-ref;ok=rel&np.isfinite(dev);t0,t1=rows[r['pair_index']]['support_seconds'];a=int(round(t0/.02));b=int(round(t1/.02));w=b-a
        fl=sustained(dev,ok,0,len(p));det=any(s<b and e>a for s,e in fl);out=any(e<=a-5 or s>=b+5 for s,e in fl)
        if not ok[a:b].any():continue
        x=P.setdefault(r['group'],dict(det=[],clean=[],med=[],w=[]));x['det'].append(det);x['clean'].append(det and not out)
        x['med'].append(float(np.median(abs(dev[a:b][ok[a:b]]))));x['w'].append(w)
    by={}
    for g in sorted(set(P)|set(G)):
        gg=G.get(g);pp=P.get(g)
        o=dict(originals=len(gg['orig']) if gg else 0,region_flag=float(np.mean(gg['reg'])) if gg and gg['reg'] else None,
            original_any_flag=float(np.mean(gg['orig'])) if gg else None,pairs=len(pp['det']) if pp else 0,
            detection=float(np.mean(pp['det'])) if pp else None,clean=float(np.mean(pp['clean'])) if pp else None)
        if pp:
            nat=[v for w in(12,14,18,25) for v in (G.get(g,{}).get('nat',{}).get(w,[]))]
            o['auroc_vs_natural']=auroc(pp['med'],nat) if nat else None
        by[g]=o
    mac=lambda k:(float(np.mean([v[k] for v in by.values() if v.get(k) is not None])) if any(v.get(k) is not None for v in by.values()) else None)
    res=dict(by_group=by,macro={k:mac(k) for k in['region_flag','original_any_flag','detection','clean','auroc_vs_natural']})
    m=res['macro'];res['silent_repair_viable']=bool(m['region_flag'] is not None and m['region_flag']<=.10 and m['detection']>=.80 and m['clean']>=.60)
    run.write(OUT/'results.json',dict(res,declaration_sha256=run.sha(OUT/'declaration.json'),no_fit=True,standin_reference=True))
    print(json.dumps(m),res['silent_repair_viable'],len(by))
main()
