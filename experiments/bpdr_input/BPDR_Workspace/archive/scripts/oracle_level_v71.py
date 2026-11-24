"""V71: oracle-span level estimator (no fit). If the TRUE span is known, can input pitch alone give the correction?
Estimator: correction = -(median pitch in span - median of flanks), flanks = 7 reliable frames each side (150ms), clipped to +-30.
Null: same estimator on every 12-frame span (stride 5) of unaltered originals; how often does natural movement look like a >=12c error?
Rule (declared before scores): viable iff singer-macro oracle MAE<=12c AND singer-macro null share(|est|>=12c)<=0.20."""
import sys,json
import numpy as np
import compare_tail_preservation as run
OUT=run.WS/'results/transformer_usability/2026-11-04_v71_oracle_level'
FL=7;W=12;ACT=12.
def est(pitch,m,a,b):
    l=slice(a-FL,a);r=slice(b,b+FL)
    if a-FL<0 or b+FL>len(pitch) or not(m[a-FL:b+FL].all()) or np.any(abs(np.diff(pitch[a-FL:b+FL]))>=70.):return None
    return float(np.clip(-(np.median(pitch[a:b])-np.median(np.r_[pitch[l],pitch[r]])),-30,30))
def main():
    if sys.argv[1:]==['declare']:
        run.write(OUT/'declaration.json',dict(doc=__doc__,tool_sha256=run.sha(__file__)));print('DECLARED');return
    pass
    d,rows,originals=run.declare();data=run.paired.records(rows,originals);G={}
    for r in data:
        g=G.setdefault(r['group'],dict(mae=[],zero=[],null=[]));p=r['raw'][:,64];m=r['raw'][:,66]>.5
        if r['kind']=='original':
            for a in range(FL,len(p)-W-FL,5):
                e=est(p,m,a,a+W)
                if e is not None:g['null'].append(abs(e))
        else:
            pos=np.flatnonzero(r['mask']&(r['target']!=0))
            if not len(pos) or np.any(np.diff(pos)>1):continue
            e=est(p,m,pos[0],pos[-1]+1)
            if e is None:continue
            t=r['target'][pos];g['mae'].append(float(np.mean(abs(e-t))));g['zero'].append(float(np.mean(abs(t))))
    by={k:dict(pairs=len(v['mae']),oracle_mae=float(np.mean(v['mae'])) if v['mae'] else None,zero_mae=float(np.mean(v['zero'])) if v['zero'] else None,
        null_spans=len(v['null']),null_share_ge12=float(np.mean(np.array(v['null'])>=ACT)) if v['null'] else None,null_median=float(np.median(v['null'])) if v['null'] else None) for k,v in G.items()}
    mac=lambda k:float(np.mean([x[k] for x in by.values() if x[k] is not None]))
    res=dict(by_group=by,macro={k:mac(k) for k in['oracle_mae','zero_mae','null_share_ge12','null_median']})
    res['viable']=bool(res['macro']['oracle_mae']<=12 and res['macro']['null_share_ge12']<=.2)
    run.write(OUT/"results.json",dict(res,declaration_sha256=run.sha(OUT/'declaration.json'),no_fit=True))
    print(json.dumps(res['macro']),res['viable'])
main()
