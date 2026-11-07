"""V74 stage 1 of the reference-conditioned review design. No fit, no new data/labels.
Reference = the unshifted original's analysis pitch (known by construction, frame-aligned with the injected clip).
Deviation(t)=pitch_injected(t)-pitch_reference(t) on reliable frames of both. Compare to injected shift = -target on plateau frames.
Flag span iff >=12 reliable plateau-region frames... implemented as: median |deviation| over the plateau (target!=0) frames >= 12c.
Metrics (singer-macro, per group kept): plateau MAE |dev-(-target)|; flag rate for pairs with |injected|>=20c (expected flagged) and
for |injected|=10c (reported only); known-zero frames with |dev|>=12c (false flag frame share); frames lost to unreliability.
Rule (declared before scores): stage passes iff macro plateau MAE<=10c AND flag rate (|inj|>=20c)>=0.90 AND false-flag frame share<=0.05.
Caveat: reference is exact by construction; originals-vs-own-reference false flags are identically zero and not reported as evidence."""
import sys,json
import numpy as np
import compare_tail_preservation as run
OUT=run.WS/'results/transformer_usability/2026-11-04_v74_deviation_report'
def main():
    if sys.argv[1:]==['declare']:
        run.write(OUT/'declaration.json',dict(doc=__doc__,tool_sha256=run.sha(__file__)));print('DECLARED');return
    assert run.read(OUT/'declaration.json')['tool_sha256']==run.sha(__file__)
    d,rows,originals=run.declare();data=run.paired.records(rows,originals)
    ref={}
    for o in originals:
        if 'excluded' in o:continue
        with np.load(o['analysis_A']) as z:ref[o['source_id']]=(z['raw_views'][0][:,64].copy(),z['raw_views'][0][:,66]>.5)
    G={};skipped=0
    for r in data:
        if r['kind']!='corrupted':continue
        src=r['source_id'].split('__')[0] if r['source_id'] not in ref else r['source_id']
        if r['source_id'] not in ref:skipped+=1;continue
        pa,ra=ref[r['source_id']];pc=r['raw'][:,64];rc=r['raw'][:,66]>.5
        if len(pa)!=len(pc):skipped+=1;continue
        both=ra&rc;dev=pc-pa;t=r['target'];pos=r['mask']&(t!=0)&both;zero=r['mask']&(t==0)&both
        if not pos.any():skipped+=1;continue
        inj=-t[pos];mae=float(np.mean(abs(dev[pos]-inj)));flag=bool(np.median(abs(dev[pos]))>=12.)
        g=G.setdefault(r['group'],dict(mae=[],flag20=[],flag10=[],fz=[],lost=[]));g['mae'].append(mae)
        (g['flag20'] if np.median(abs(inj))>=20 else g['flag10']).append(flag)
        g['fz'].append(float(np.mean(abs(dev[zero])>=12.)) if zero.any() else np.nan)
        g['lost'].append(1-pos.sum()/max(1,(r['mask']&(t!=0)).sum()))
    by={k:dict(pairs=len(v['mae']),mae=float(np.mean(v['mae'])),flag_rate_ge20=float(np.mean(v['flag20'])) if v['flag20'] else None,
        flag_rate_10=float(np.mean(v['flag10'])) if v['flag10'] else None,false_flag_frame_share=(float(np.nanmean(v['fz'])) if np.isfinite(v['fz']).any() else None),plateau_frames_lost=float(np.mean(v['lost']))) for k,v in G.items()}
    m=lambda k:(float(np.mean([x[k] for x in by.values() if x[k] is not None and np.isfinite(x[k])])) if any(x[k] is not None and np.isfinite(x[k]) for x in by.values()) else None)
    res=dict(by_group=by,macro={k:m(k) for k in['mae','flag_rate_ge20','flag_rate_10','false_flag_frame_share','plateau_frames_lost']},pairs_skipped=skipped)
    res['passes']=bool(res['macro']['mae']<=10 and res['macro']['flag_rate_ge20']>=.9 and res['macro']['false_flag_frame_share']<=.05)
    run.write(OUT/'results.json',dict(res,declaration_sha256=run.sha(OUT/'declaration.json'),no_fit=True))
    print(json.dumps(res['macro']),res['pairs_skipped'],res['passes'],len(by))
main()
