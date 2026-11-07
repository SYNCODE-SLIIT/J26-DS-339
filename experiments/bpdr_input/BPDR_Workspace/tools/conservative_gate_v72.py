"""V72: abstain-by-default gate on existing held predictions (V68). No fit, no new data.
Candidate = run of >=MIN_FRAMES consecutive reliable frames, same sign, median-filtered(3) |pred|>=T, none within protected guard (jump>=70c / energy / reliability edges, 4 frames).
PRIMARY point declared before scores: T=24c, MIN_FRAMES=12 (.24s). T=12 and 18 reported descriptively only; no selection among them.
Per original: abstained if no candidate. Per pair: clean_hit = >=1 candidate overlapping injection AND no candidate span touching known-zero reliable frames outside injection; false_edit = any candidate touching known-zero frames.
Rule (primary, singer-macro for pairs, original-level for originals): viable iff originals_with_candidate<=0.05 AND clean_hit>=0.25 AND false_edit<=0.10."""
import sys,json
import numpy as np
from scipy.ndimage import median_filter
import compare_tail_preservation as run
OUT=run.WS/'results/transformer_usability/2026-11-04_v72_conservative_gate'
MIN=12;PRIMARY=24.
def protected(raw):
    rel=raw[:,66]>.5;ch=np.r_[True,np.diff(rel.astype(int))!=0]
    j=np.r_[False,(abs(np.diff(raw[:,64]))>=70)&rel[1:]&rel[:-1]];e=np.r_[False,(abs(np.diff(raw[:,68]))>=6)&rel[1:]&rel[:-1]]
    pr=np.zeros(len(raw),bool)
    for i in np.flatnonzero(ch|j|e):pr[max(0,i-4):i+5]=True
    return pr,rel
def cands(pred,raw,T):
    pr,rel=protected(raw);f=median_filter(pred,size=3,mode='nearest');out=[]
    for s in(-1,1):
        on=(f*s>=T)&rel;i=0
        while i<len(on):
            if on[i]:
                j=i
                while j<len(on) and on[j]:j+=1
                if j-i>=MIN and not pr[i:j].any():out.append((i,j))
                i=j
            else:i+=1
    return out
def main():
    if sys.argv[1:]==['declare']:
        run.write(OUT/'declaration.json',dict(doc=__doc__,tool_sha256=run.sha(__file__)));print('DECLARED');return
    assert run.read(OUT/'declaration.json')['tool_sha256']==run.sha(__file__)
    d,rows,originals=run.declare();data=run.paired.records(rows,originals);res={}
    for arm in run.ARMS:
        with np.load(run.ROOT/(arm+'_all_predictions.npz')) as z:pred={k:z[k].copy() for k in z.files}
        res[arm]={}
        for T in(12.,18.,24.):
            og=[];G={}
            for r in data:
                c=cands(pred[r['id']],r['raw'],T)
                if r['kind']=='original':og.append(bool(c));continue
                pos=r['mask']&(r['target']!=0);zero=r['mask']&(r['target']==0)
                if not pos.any():continue
                hit=any(pos[a:b].any() for a,b in c);fe=any(zero[a:b].any() for a,b in c)
                x=G.setdefault(r['group'],[]);x.append((hit and not fe,fe,bool(c)))
            m=lambda k:float(np.mean([np.mean([v[k] for v in L]) for L in G.values()]))
            res[arm][str(T)]=dict(originals=len(og),originals_with_candidate=float(np.mean(og)),clean_hit=m(0),false_edit=m(1),any_candidate=m(2),groups=len(G),
                viable=bool(np.mean(og)<=.05 and m(0)>=.25 and m(1)<=.10),primary=T==PRIMARY)
    run.write(OUT/'results.json',dict(arms=res,declaration_sha256=run.sha(OUT/'declaration.json'),no_fit=True))
    print(json.dumps(res,indent=1))
main()
