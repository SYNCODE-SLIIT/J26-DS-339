"""V73: does a step-shape measure separate injected spans from natural movement? No fit, no new data.
Score(span [a,b)) = min(|entry|,|exit|) if entry*exit>0 else 0 (cents), entry=median(p[a+1:a+4])-median(p[a-4:a-1]), exit=median(p[b-4:b-1])-median(p[b+1:b+4]),
p = existing feature pitch (raw[:,64]); span+7-frame flanks must be reliable with no >=70c jump (same steady-region rule as V71).
Positives: true injected span of each valid pair. Nulls: all spans of the SAME width (12,14,18,25 frames) at stride 3 in unaltered originals (pooled across groups; originals are the same source recordings).
Metric: singer-macro AUROC(positive score > pooled null score) per width-matched null; comparison metric: level |V71 estimate| same way.
Rule (declared before scores): viable iff macro AUROC(step) >= 0.85. Otherwise: pitch-trace shape cannot separate; richer audio evidence needs a separate declared study."""
import sys,json
import numpy as np
import compare_tail_preservation as run
OUT=run.WS/'results/transformer_usability/2026-11-04_v73_edge_shape'
FL=7;WIDTHS=(12,14,18,25)
def scores(p,rel,a,b):
    if a-FL<0 or b+FL>len(p) or not rel[a-FL:b+FL].all() or np.any(abs(np.diff(p[a-FL:b+FL]))>=70.):return None
    med=np.median;en=med(p[a+1:a+4])-med(p[a-4:a-1]);ex=med(p[b-4:b-1])-med(p[b+1:b+4])
    step=min(abs(en),abs(ex)) if en*ex>0 else 0.
    lvl=abs(med(p[a:b])-med(np.r_[p[a-FL:a],p[b:b+FL]]))
    return step,lvl
def auroc(a,b):
    a=np.asarray(a);b=np.asarray(b);return float(np.mean((a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])))
def main():
    if sys.argv[1:]==['declare']:
        run.write(OUT/'declaration.json',dict(doc=__doc__,tool_sha256=run.sha(__file__)));print('DECLARED');return
    assert run.read(OUT/'declaration.json')['tool_sha256']==run.sha(__file__)
    d,rows,originals=run.declare();data=run.paired.records(rows,originals)
    null={w:[] for w in WIDTHS};pos={}
    for r in data:
        p=r['raw'][:,64];rel=r['raw'][:,66]>.5
        if r['kind']=='original':
            for w in WIDTHS:
                for a in range(FL,len(p)-w-FL,3):
                    s=scores(p,rel,a,a+w)
                    if s:null[w].append(s)
        else:
            t0,t1=rows[r['pair_index']]['support_seconds'];a=int(round(t0/.02));b=int(round(t1/.02));w=b-a
            idx=np.flatnonzero(r['mask']&(r['target']!=0))
            if w not in null or not len(idx) or idx[0]<a or idx[-1]>=b:continue
            s=scores(p,rel,a,b)
            if s:pos.setdefault(r['group'],[]).append((w,)+s)
    by={}
    for g,L in pos.items():
        o=dict(pairs=len(L))
        for k,name in((1,'step'),(2,'level')):
            vals=[auroc([x[k]],[n[k-1] for n in null[x[0]]]) for x in L if null[x[0]]]
            o['auroc_'+name]=float(np.mean(vals)) if vals else None
        by[g]=o
    mac=lambda k:float(np.mean([v[k] for v in by.values() if v[k] is not None]))
    res=dict(by_group=by,macro=dict(auroc_step=mac('auroc_step'),auroc_level=mac('auroc_level')),null_spans={str(w):len(v) for w,v in null.items()})
    res['viable']=bool(res['macro']['auroc_step']>=.85)
    run.write(OUT/'results.json',dict(res,declaration_sha256=run.sha(OUT/'declaration.json'),no_fit=True))
    print(json.dumps(res['macro']),res['null_spans'],res['viable'],len(by))
main()
