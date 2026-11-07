"""V70: does the existing held-out prediction separate the injection from the rest of the clip?

No fitting, no new data/labels. Reads V65/V68 held predictions. Rule is written to
declaration.json (with this file's hash) BEFORE any score is computed.
"""
from pathlib import Path
import json,sys
import numpy as np
import compare_tail_preservation as run

OUT=run.WS/'results/transformer_usability/2026-11-04_v70_detection_diagnosis'
ACT=12.  # existing registered ShortPolicy.activation_cents; not tuned
RULE=dict(version='detection-diagnosis-1',activation_cents=ACT,
    pair_score_in='max over injected frames of prediction*sign(target)',
    pair_score_zero='max |prediction| over reliable known-zero frames of same clip',
    pair_score_orig='max |prediction| over reliable frames of the matched original source (window)',
    metrics=['paired_auroc_in_vs_zero','paired_auroc_in_vs_orig','clean_hit(in>=ACT and zero<ACT)','orig_quiet(orig<ACT)','argmax_in_injection'],
    aggregation='singer-macro over groups with scoreable pairs; per-group values retained',
    decision_rule=dict(detection_adequate_if='macro paired_auroc_in_vs_zero>=0.85 AND macro clean_hit>=0.50 AND macro orig_quiet>=0.90',
        else_branch='detection_weak: stop Transformer tuning; remaining levers are region availability and BPDR/aug/frame objective comparison'),
    no_threshold_selection=True,arms=run.ARMS)

def auroc(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return float(np.mean((a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])))

def main():
    if sys.argv[1:]==['declare']:
        d=dict(RULE,tool_sha256=run.sha(__file__));run.write(OUT/'declaration.json',d);print('DECLARED');return
    decl=run.read(OUT/'declaration.json');assert decl['tool_sha256']==run.sha(__file__)
    d,rows,originals=run.declare();data=run.paired.records(rows,originals)
    orig_by_source={}
    out={}
    for arm in run.ARMS:
        with np.load(run.ROOT/(arm+'_all_predictions.npz')) as z:pred={k:z[k].copy() for k in z.files}
        orig={}
        for r in data:
            if r['kind']=='original':
                p=np.abs(pred[r['id']][r['mask']]);orig[r['source_id']]=float(p.max()) if len(p) else None
        recs=[]
        for r in data:
            if r['kind']!='corrupted':continue
            p=pred[r['id']];m=r['mask'];t=r['target'];pos=m&(t!=0);zero=m&(t==0)
            if not pos.any() or not zero.any():continue
            s_in=float(np.max(p[pos]*np.sign(t[pos])));s_zero=float(np.max(abs(p[zero])))
            a=np.where(m,abs(p),-1);recs.append(dict(id=r['id'],group=r['group'],source_id=r['source_id'],s_in=s_in,s_zero=s_zero,
                s_orig=orig.get(r['source_id']),argmax_in=bool(pos[int(np.argmax(a))]),
                clean=bool(s_in>=ACT and s_zero<ACT)))
        groups={}
        for g in sorted({x['group'] for x in recs}):
            R=[x for x in recs if x['group']==g]
            ins=[x['s_in'] for x in R];zs=[x['s_zero'] for x in R]
            os_=[x['s_orig'] for x in R if x['s_orig'] is not None]
            allo=[v for v in orig.values() if v is not None]
            groups[g]=dict(pairs=len(R),paired_auroc_in_vs_zero=float(np.mean([x['s_in']>x['s_zero'] for x in R])),
                auroc_in_vs_orig=auroc(ins,os_) if os_ else None,clean_hit=float(np.mean([x['clean'] for x in R])),
                in_ge_act=float(np.mean([s>=ACT for s in ins])),zero_ge_act=float(np.mean([s>=ACT for s in zs])),
                orig_quiet=float(np.mean([v<ACT for v in set(os_)])) if os_ else None,
                argmax_in_injection=float(np.mean([x['argmax_in'] for x in R])),median_s_in=float(np.median(ins)),median_s_zero=float(np.median(zs)))
        mac=lambda k:float(np.mean([v[k] for v in groups.values() if v[k] is not None]))
        macro={k:mac(k) for k in ['paired_auroc_in_vs_zero','auroc_in_vs_orig','clean_hit','in_ge_act','zero_ge_act','orig_quiet','argmax_in_injection']}
        macro['detection_adequate']=bool(macro['paired_auroc_in_vs_zero']>=.85 and macro['clean_hit']>=.5 and macro['orig_quiet']>=.9)
        out[arm]=dict(macro=macro,by_group=groups,pairs=len(recs))
    run.write(OUT/'results.json',dict(declaration_sha256=run.sha(OUT/'declaration.json'),arms=out,no_fit=True,production_eligible=False))
    print(json.dumps({a:v['macro'] for a,v in out.items()},indent=1))

if __name__=='__main__':main()
