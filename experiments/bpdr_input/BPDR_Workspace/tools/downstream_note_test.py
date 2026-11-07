"""Downstream note test with a STAND-IN extractor: the team's melody_understanding pYIN baseline (nearest-semitone notes). No fit.
Question: does BPDR-delivered audio change the extractor's note labels back toward the unaltered original's labels?
Windows: [support_start-1.0s, support_end+1.0s] cropped from (O) canonical original, (C) injected clip, (D) delivered V76 audio (verified candidate if any, else the injected clip: intent-to-treat).
Per-frame label = nearest MIDI note of the extractor's notes on a 20 ms grid (rest=-1). Agreement(X) = share of frames where label(X)==label(O), computed IN the injected span and OUTSIDE it (window minus span, +-0.1s guard).
Pair gain = agreement_in(D)-agreement_in(C); damage = agreement_out(C)-agreement_out(D).
Singer-macro over groups with scoreable pairs. Rule (declared before scores): delivered audio helps downstream iff macro gain>0 with a paired singer bootstrap 95% interval above 0
AND macro damage<=0.005. Also report share of pairs where the injection changed any in-span label (the achievable ceiling). Oracle-reference repair (V76): an upper bound, not a deployable result.
Limits: extractor is a deterministic pYIN baseline, not the team's final model; reference labels are the extractor's own labels on the original, not human notes."""
import sys,json,tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import compare_tail_preservation as run
sys.path.insert(0,str(run.WS.parents[2]/'src'))
from melody_understanding.pipeline import analyze_vocal
OUT=run.WS/'results/transformer_usability/2026-11-05_downstream_note_test'
V76=run.WS/'results/transformer_usability/2026-11-05_v76_applied_shift_cap'
STEP=.02
def labels(path,n,t0):
    r,_=analyze_vocal(str(path));lab=np.full(n,-1,int);t=t0+np.arange(n)*STEP
    for nt in r['notes']:
        lab[(t-t0>=nt['onset_s'])&(t-t0<nt['onset_s']+nt['duration_s'])]=nt['pitch_midi']
    return lab
def main():
    if sys.argv[1:]==['declare']:
        run.write(OUT/'declaration.json',dict(doc=__doc__,tool_sha256=run.sha(__file__)));print('DECLARED');return
    assert run.read(OUT/'declaration.json')['tool_sha256']==run.sha(__file__)
    d,rows,originals=run.declare();by={s['source_id']:s for s in originals if 'excluded' not in s}
    v76={r['id']:r for r in run.read(V76/'results.json')['records']};tmp=Path(tempfile.mkdtemp(dir=OUT));recs=[]
    for i,row in enumerate(rows):
        if 'excluded' in row:continue
        o=by[row['source_id']];t0,t1=row['support_seconds'];x,sr=sf.read(o['path_A'],dtype='float32')
        c,_=sf.read(row['path_C'],dtype='float32');cid='C_'+str(i)
        dl=c
        if v76[cid]['verified']:dl,_=sf.read(V76/cid/'candidate.wav',dtype='float32')
        a=max(0,int((t0-1.)*sr));b=min(len(x),int((t1+1.)*sr));L=min(len(x),len(c),len(dl))
        if b>L:b=L
        n=int(np.ceil((b-a)/sr/STEP));lab={}
        for k,sig in(('O',x),('C',c),('D',dl)):
            p=tmp/f'{k}.wav';sf.write(p,sig[a:b],sr);lab[k]=labels(p,n,0.)
        t=(a/sr)+np.arange(n)*STEP;ins=(t>=t0)&(t<t1);out=(t<t0-.1)|(t>=t1+.1)
        if not ins.any() or not out.any():continue
        ag=lambda k,m:float(np.mean(lab[k][m]==lab['O'][m]))
        recs.append(dict(id=cid,group=row['group'],width=row['width_seconds'],cents=row['cents'],in_C=ag('C',ins),in_D=ag('D',ins),out_C=ag('C',out),out_D=ag('D',out),
            injection_changed_label=bool(ag('C',ins)<1.),delivered=bool(v76[cid]['verified'])))
        if len(recs)%25==0:print('progress',len(recs),flush=True)
    groups=sorted({r['group'] for r in recs})
    gm=lambda f:np.array([np.mean([f(r) for r in recs if r['group']==g]) for g in groups])
    gain=gm(lambda r:r['in_D']-r['in_C']);dam=gm(lambda r:r['out_C']-r['out_D']);ceil=gm(lambda r:r['injection_changed_label'])
    rng=np.random.default_rng(339);boot=gain[rng.integers(0,len(gain),size=(4000,len(gain)))].mean(axis=1)
    res=dict(pairs=len(recs),groups=len(groups),macro_in_C=float(gm(lambda r:r['in_C']).mean()),macro_in_D=float(gm(lambda r:r['in_D']).mean()),
        macro_gain=float(gain.mean()),gain_ci95=np.quantile(boot,[.025,.975]).tolist(),macro_damage=float(dam.mean()),macro_injection_changed_label_share=float(ceil.mean()),
        delivered_share=float(np.mean([r['delivered'] for r in recs])))
    res['helps']=bool(res['macro_gain']>0 and res['gain_ci95'][0]>0 and res['macro_damage']<=.005)
    run.write(OUT/'results.json',dict(res,records=recs,declaration_sha256=run.sha(OUT/'declaration.json'),no_fit=True,oracle_reference=True,standin_extractor='melody_understanding pyin_baseline'))
    print('DOWNSTREAM',json.dumps(res),flush=True)
main()
