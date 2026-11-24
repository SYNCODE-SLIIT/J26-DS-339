"""Restoration + closed-loop gate test. No fit. Clean base: 53 working windows (first 4 s), 17 singer groups.
Corruptions (synthetic, from damage_map.py): white/pink noise 10,0 dB; clip .3,.1; gain -60,-70 dB.
Operators (own classical implementations, no new dependencies; all thresholds declared here, none tuned after scores):
 L level: if RMS < -35 dBFS scale to -23 dBFS RMS, peak <= .97.
 D declip: if >=0.1% of samples sit in runs (>=2) at >=0.995*peak, replace those runs by a cubic spline through 6 neighbouring unclipped samples each side, kept beyond the clip level.
 N denoise: if est. SNR = 10log10(p90/p10 of 20 ms frame energies) < 25 dB, STFT (1024/256) Wiener-style gate, noise profile = 10th-percentile magnitude per bin, gain floor -20 dB.
Variants: C corrupted; R restored (L,D,N applied by the detectors above, input-only);
 G gated = R only if cross-extractor consensus(R) >= consensus(C)+0.02 (consensus = frames where pYIN, PESTO, Praat all give the same non-rest label / frames where any is voiced;
 input-only, no clean reference), else C; W = 0.5C+0.5R dry/wet control.
Scoring uses the clean window ONLY after delivery: label_recall of each frozen extractor vs its own clean output (as damage_map.py). Harm = recall drop > 0.02 vs C.
Rule (declared before scores): the gate is useful iff, over the 8 corruptions, singer-macro recall(G)-recall(C) > 0 with singer-bootstrap 95% lower bound > 0 for ALL THREE extractors
 AND the gated harm rate <= 0.05 per extractor. Always-restore (R) and oracle gate (best of C,R per clip/extractor) reported for comparison; W reported for PESTO and Praat only (cost).
Limits: synthetic corruptions; self-referenced labels; consensus proxy uses the same three trackers as scoring (so improvement may be partly circular); not human labels."""
import sys,json
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.interpolate import CubicSpline
import compare_tail_preservation as run
OUT=run.WS/'results/transformer_usability/2026-11-05_restoration_gate'
ns={}
exec(open(run.WS/'tools/damage_map.py',encoding='utf8').read().rsplit('\nmain()',1)[0],ns)
SR=ns['SR'];STEP=ns['STEP'];corrupt=ns['corrupt'];EXT=ns['EXT'];SEC4=4
CONDS=[('white',10),('white',0),('pink',10),('pink',0),('clip',.3),('clip',.1),('gain',-60),('gain',-70)]


def level(y):
    r=np.sqrt(np.mean(y**2))+1e-12
    if 20*np.log10(r)<-35:
        g=10**(-23/20)/r;g=min(g,.97/(np.max(abs(y))+1e-12));return y*g
    return y


def declip(y):
    pk=np.max(abs(y));m=abs(y)>=.995*pk;runs=[];i=0
    while i<len(y):
        if m[i]:
            j=i
            while j<len(y) and m[j]:j+=1
            if j-i>=2:runs.append((i,j))
            i=j
        else:i+=1
    if sum(b-a for a,b in runs)<.001*len(y):return y
    z=y.copy();bad=np.zeros(len(y),bool)
    for a,b in runs:bad[a:b]=True
    for a,b in runs:
        idx=np.r_[np.arange(max(0,a-6),a),np.arange(b,min(len(y),b+6))];idx=idx[~bad[idx]]
        if len(idx)<4:continue
        seg=CubicSpline(idx,y[idx])(np.arange(a,b));s=np.sign(y[a]);z[a:b]=s*np.maximum(s*seg,pk)
    return z


def denoise(y):
    fl=int(.02*SR);n=len(y)//fl
    if n<5:return y
    e=np.array([np.mean(y[i*fl:(i+1)*fl]**2) for i in range(n)])+1e-12
    if 10*np.log10(np.percentile(e,90)/np.percentile(e,10))>=25:return y
    f,t,Z=signal.stft(y,SR,nperseg=1024,noverlap=768);M=abs(Z);N=np.percentile(M,10,axis=1,keepdims=True)
    G=np.maximum(1-(N/(M+1e-12))**2,10**(-20/20))
    _,x=signal.istft(Z*G,SR,nperseg=1024,noverlap=768);x=x[:len(y)]
    return np.pad(x,(0,len(y)-len(x)))


def restore(y):return denoise(declip(level(y)))


def consensus(L):
    a,b,c=L;any_=(a!=-1)|(b!=-1)|(c!=-1);ag=(a==b)&(b==c)&(a!=-1)
    return float(ag.sum()/max(1,any_.sum()))


def main():
    if sys.argv[1:]==['declare']:
        run.write(OUT/'declaration.json',dict(doc=__doc__,tool_sha256=run.sha(__file__)));print('DECLARED');return
    assert run.read(OUT/'declaration.json')['tool_sha256']==run.sha(__file__)
    d,rows,originals=run.declare();rng=np.random.default_rng(339);recs=[]
    for k,o in enumerate([o for o in originals if 'excluded' not in o]):
        x,sr=sf.read(o['path_A'],dtype='float32');x=x[:SEC4*SR].astype(np.float64);n=int(len(x)/SR/STEP)
        clean={e:f(x.astype(np.float32),n) for e,f in EXT.items()}
        for kind,v in CONDS:
            y=corrupt(x,kind,v,rng);R=restore(y);W=.5*y+.5*R;lab={}
            for nm,sig in(('C',y),('R',R),('W',W)):
                for e,f in EXT.items():
                    if nm=='W' and e=='pyin':continue
                    lab[(nm,e)]=f(sig.astype(np.float32),n)
            cC=consensus([lab[('C',e)] for e in EXT]);cR=consensus([lab[('R',e)] for e in EXT]);acc=bool(cR>=cC+.02)
            for e in EXT:
                cv=clean[e]!=-1
                rec=lambda L:float(np.mean(L[cv]==clean[e][cv])) if cv.any() else None
                recs.append(dict(group=o['group'],src=o['source_id'],kind=kind,level=v,ext=e,C=rec(lab[('C',e)]),R=rec(lab[('R',e)]),
                    W=rec(lab[('W',e)]) if ('W',e) in lab else None,accepted=acc,cons_C=cC,cons_R=cR))
        print('progress',k+1,flush=True)
    recs=[r for r in recs if r['C'] is not None]
    for r in recs:
        r['G']=r['R'] if r['accepted'] else r['C'];r['O']=max(r['C'],r['R'])
    out={};ok=True
    for e in EXT:
        R_=[r for r in recs if r['ext']==e];g=sorted({r['group'] for r in R_})
        gm=lambda f:np.array([np.mean([f(r) for r in R_ if r['group']==x]) for x in g])
        gain=gm(lambda r:r['G']-r['C']);boot=gain[np.random.default_rng(339).integers(0,len(g),size=(4000,len(g)))].mean(axis=1);lo=float(np.quantile(boot,.025))
        d_=dict(macro_C=float(gm(lambda r:r['C']).mean()),macro_R=float(gm(lambda r:r['R']).mean()),macro_G=float(gm(lambda r:r['G']).mean()),macro_oracle=float(gm(lambda r:r['O']).mean()),
            gain_G=float(gain.mean()),gain_G_ci_lo=lo,harm_R=float(np.mean([r['R']<r['C']-.02 for r in R_])),harm_G=float(np.mean([r['G']<r['C']-.02 for r in R_])),
            accept_rate=float(np.mean([r['accepted'] for r in R_])),
            macro_W=float(gm(lambda r:r['W'] if r['W'] is not None else r['C']).mean()) if e!='pyin' else None)
        out[e]=d_;ok&=bool(d_['gain_G']>0 and lo>0 and d_['harm_G']<=.05)
    by={}
    for kind,v in CONDS:
        for e in EXT:
            L=[r for r in recs if r['ext']==e and r['kind']==kind and r['level']==v]
            by[f'{e}|{kind}|{v}']=dict(C=float(np.mean([r['C'] for r in L])),R=float(np.mean([r['R'] for r in L])),G=float(np.mean([r['G'] for r in L])),accept=float(np.mean([r['accepted'] for r in L])))
    run.write(OUT/'results.json',dict(summary=out,by_condition=by,gate_useful=ok,declaration_sha256=run.sha(OUT/'declaration.json'),no_fit=True,synthetic=True))
    print('GATE',json.dumps(dict(summary=out,gate_useful=ok)),flush=True)


main()
