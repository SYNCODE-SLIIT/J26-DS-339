"""Restoration gate, round 2 (adds heavy-clipping REFUSAL and a classical dereverberator). No fit. Same clean base/trackers/scoring as restoration_gate.py
(53 windows, first 4 s, 17 groups; label recall vs each tracker's own clean output). Conditions run here: clip .3, clip .1, reverb RT60 .4, reverb RT60 .8 (others unchanged from round 1, whose failed
declared rule stays on record).
New operators (thresholds declared before scores, none tuned after):
 REFUSE: if >=2% of samples sit in runs (>=2) at >=0.995*peak -> output = unchanged input, status REFUSED with time ranges of the clipped runs (re-record guidance). Not counted as restored.
 E dereverb (Lebart-style late-reverberation spectral subtraction): blind RT60 from broadband STFT energy (1024/256): median slope of decreasing runs (>=6 frames, >=15 dB drop); apply only if RT60 > 0.30 s;
   late power P_late(t,f) = exp(-13.8*0.046/RT60) * P(t-4,f); gain = sqrt(max(1-P_late/P, 0.03)).
 Pipeline R2: level -> declip (only if not refused) -> denoise -> dereverb. Gate unchanged: accept R2 only if 3-tracker consensus rises >= 0.02 (input-only).
Rule (declared before scores): (i) gated harm rate (recall drop > 0.02 vs unprocessed) <= 0.05 for every tracker in every condition group {clip, reverb};
 (ii) reverb family: gated singer-macro gain over unprocessed has bootstrap 95% lower bound > 0 for >= 2 of 3 trackers; (iii) refusal: report refusal rate per condition, recall with refused clips counted as unprocessed (intention-to-treat),
 and for the accepted-only subset. Also report the round-1 gated recall for the same clip conditions for comparison. Round-2 success needs (i) and (ii).
Limits: synthetic corruptions and RIRs; self-referenced labels; consensus proxy shares trackers with scoring; own implementations; not human labels."""
import sys,json
import numpy as np
import soundfile as sf
from scipy import signal
import compare_tail_preservation as run
OUT=run.WS/'results/transformer_usability/2026-11-05_restoration_gate2'
g1={}
exec(open(run.WS/'tools/restoration_gate.py',encoding='utf8').read().rsplit('\n\nmain()',1)[0],g1)
level,declip,denoise,consensus=g1['level'],g1['declip'],g1['denoise'],g1['consensus']
SR=g1['SR'];STEP=g1['STEP'];corrupt=g1['corrupt'];EXT=g1['EXT'];SEC4=4
CONDS=[('clip',.3),('clip',.1),('reverb',.4),('reverb',.8)]


def clipped_fraction(y):
    pk=np.max(abs(y));m=abs(y)>=.995*pk;runs=[];i=0
    while i<len(y):
        if m[i]:
            j=i
            while j<len(y) and m[j]:j+=1
            if j-i>=2:runs.append((i,j))
            i=j
        else:i+=1
    return sum(b-a for a,b in runs)/len(y),runs


def est_rt60(y):
    f,t,Z=signal.stft(y,SR,nperseg=1024,noverlap=768);E=10*np.log10(np.sum(abs(Z)**2,axis=0)+1e-12);hop=256/SR
    slopes=[];i=0
    while i<len(E)-1:
        j=i
        while j+1<len(E) and E[j+1]<E[j]:j+=1
        if j-i>=6 and E[i]-E[j]>=15:slopes.append((E[j]-E[i])/((j-i)*hop))
        i=max(j,i+1)
    return None if not slopes else float(-60/np.median(slopes))


def dereverb(y):
    rt=est_rt60(y)
    if rt is None or rt<=.30:return y,rt
    f,t,Z=signal.stft(y,SR,nperseg=1024,noverlap=768);P=abs(Z)**2;late=np.zeros_like(P);late[:,4:]=np.exp(-13.8*.046/rt)*P[:,:-4]
    G=np.sqrt(np.maximum(1-late/(P+1e-12),.03));_,x=signal.istft(Z*G,SR,nperseg=1024,noverlap=768);x=x[:len(y)]
    return np.pad(x,(0,len(y)-len(x))),rt


def restore2(y):
    frac,runs=clipped_fraction(y)
    if frac>=.02:return y,'REFUSED',dict(clipped_fraction=frac,runs_seconds=[(a/SR,b/SR) for a,b in runs][:50])
    z=denoise(declip(level(y)));z,rt=dereverb(z);return z,'RESTORED',dict(clipped_fraction=frac,rt60=rt)


def main():
    if sys.argv[1:]==['declare']:
        run.write(OUT/'declaration.json',dict(doc=__doc__,tool_sha256=run.sha(__file__)));print('DECLARED');return
    assert run.read(OUT/'declaration.json')['tool_sha256']==run.sha(__file__)
    d,rows,originals=run.declare();rng=np.random.default_rng(339);recs=[]
    for k,o in enumerate([o for o in originals if 'excluded' not in o]):
        x,sr=sf.read(o['path_A'],dtype='float32');x=x[:SEC4*SR].astype(np.float64);n=int(len(x)/SR/STEP)
        clean={e:f(x.astype(np.float32),n) for e,f in EXT.items()}
        for kind,v in CONDS:
            y=corrupt(x,kind,v,rng);R,status,info=restore2(y);lab={}
            for nm,sig in(('C',y),('R',R)):
                for e,f in EXT.items():lab[(nm,e)]=f(sig.astype(np.float32),n) if not(nm=='R' and status=='REFUSED') else lab[('C',e)]
            cC=consensus([lab[('C',e)] for e in EXT]);cR=consensus([lab[('R',e)] for e in EXT]);acc=bool(status=='RESTORED' and cR>=cC+.02)
            for e in EXT:
                cv=clean[e]!=-1
                rec=lambda L:float(np.mean(L[cv]==clean[e][cv])) if cv.any() else None
                recs.append(dict(group=o['group'],src=o['source_id'],kind=kind,level=v,ext=e,C=rec(lab[('C',e)]),R=rec(lab[('R',e)]),status=status,accepted=acc,
                    clipped_fraction=info['clipped_fraction'],rt60=info.get('rt60')))
        print('progress',k+1,flush=True)
    recs=[r for r in recs if r['C'] is not None]
    for r in recs:r['G']=r['R'] if r['accepted'] else r['C']
    out={};harm_ok=True;rev_hits=0
    for fam,kinds in(('clip',['clip']),('reverb',['reverb'])):
        for e in EXT:
            R_=[r for r in recs if r['ext']==e and r['kind'] in kinds];g=sorted({r['group'] for r in R_})
            gm=lambda f:np.array([np.mean([f(r) for r in R_ if r['group']==x]) for x in g])
            gain=gm(lambda r:r['G']-r['C']);boot=gain[np.random.default_rng(339).integers(0,len(g),size=(4000,len(g)))].mean(axis=1);lo=float(np.quantile(boot,.025))
            hg=float(np.mean([r['G']<r['C']-.02 for r in R_]));harm_ok&=hg<=.05
            if fam=='reverb' and lo>0:rev_hits+=1
            out[f'{fam}|{e}']=dict(macro_C=float(gm(lambda r:r['C']).mean()),macro_R=float(gm(lambda r:r['R']).mean()),macro_G=float(gm(lambda r:r['G']).mean()),gain_G=float(gain.mean()),gain_ci_lo=lo,
                harm_R=float(np.mean([r['R']<r['C']-.02 for r in R_])),harm_G=hg,accept_rate=float(np.mean([r['accepted'] for r in R_])),refused_rate=float(np.mean([r['status']=='REFUSED' for r in R_])))
    by={}
    for kind,v in CONDS:
        L=[r for r in recs if r['kind']==kind and r['level']==v]
        by[f'{kind}|{v}']=dict(refused=float(np.mean([r['status']=='REFUSED' for r in L])),accepted=float(np.mean([r['accepted'] for r in L])),
            clipped_fraction_median=float(np.median([r['clipped_fraction'] for r in L])),rt60_median=float(np.median([r['rt60'] for r in L if r['rt60'] is not None])) if any(r['rt60'] is not None for r in L) else None,
            recall={e:dict(C=float(np.mean([r['C'] for r in L if r['ext']==e])),G=float(np.mean([r['G'] for r in L if r['ext']==e]))) for e in EXT})
    res=dict(summary=out,by_condition=by,harm_ok=bool(harm_ok),reverb_extractors_with_gain=int(rev_hits),round2_success=bool(harm_ok and rev_hits>=2))
    run.write(OUT/'results.json',dict(res,declaration_sha256=run.sha(OUT/'declaration.json'),no_fit=True,synthetic=True))
    print('GATE2',json.dumps(res),flush=True)


main()
