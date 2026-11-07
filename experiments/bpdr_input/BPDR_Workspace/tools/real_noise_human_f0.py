"""Real-data test: real recorded noise (ESC-50 root clips: rain, vacuum cleaner, breathing, clock tick) mixed into the 21 training-role Vocadito recordings (real phone/computer singing,
human F0 annotations), scored against the HUMAN F0 annotation (not self-referenced). No fit. First 8 s of each recording. The 4 ESC-50 'validation' clips are NOT used (closed reserve).
SNR {10, 0} dB relative to overall RMS, noise looped from a fixed offset. Frozen trackers: pYIN (team params), PESTO 2.0.1 (conf>0.5), Praat-AC (75-800 Hz, 10 ms).
Metric: RPA50 = share of human-voiced reference frames where the tracker is voiced and within 50 cents (nearest tracker frame). Singer groups (14) are the unit of statistics.
Variants: clean; C noisy; R restored (level -> declip -> spectral-gate denoise, same detectors/thresholds as restoration_gate.py); G gated (R only if 3-tracker label consensus rises >= 0.02, input-only).
Rule (declared before scores): (i) gated per-recording RPA drop > 0.02 vs C occurs in <= 5% of cases for every tracker; (ii) at 0 dB, singer-macro RPA gain G-C has a bootstrap 95% lower bound > 0 for >= 2 of 3 trackers.
Also reported: clean-RPA ceiling and the share of noise-induced loss recovered (G-C)/(clean-C); R (always restore) for comparison.
Limits: ESC-50 is CC BY-NC (research use); only 4 noise clips, 21 recordings; human F0 annotations are performed pitch not intended pitch; consensus proxy shares trackers with scoring; not replayed."""
import sys,json,csv
import numpy as np
import soundfile as sf
from scipy import signal
import compare_tail_preservation as run
OUT=run.WS/'results/transformer_usability/2026-11-05_real_noise_human_f0'
g1={}
exec(open(run.WS/'tools/restoration_gate.py',encoding='utf8').read().rsplit('\n\nmain()',1)[0],g1)
restore,consensus=g1['restore'],g1['consensus']
SR=22050;STEP=.02;SEC=8
IDS=[1,10,13,14,18,20,25,26,27,28,29,3,31,32,35,36,37,38,39,6,8]
NOISE=['1-17367-A-10.wav','1-100210-A-36.wav','1-18631-A-23.wav','1-21934-A-38.wav']
VOC=run.WS/'datasets/vocadito'


def load(path):
    x,sr=sf.read(path,dtype='float32')
    if x.ndim>1:x=x.mean(axis=1)
    if sr!=SR:
        from math import gcd
        g=gcd(sr,SR);x=signal.resample_poly(x,SR//g,sr//g)
    return x.astype(np.float64)


def hz_pyin(x):
    import librosa
    f0,v,_=librosa.pyin(x,fmin=librosa.note_to_hz('C1'),fmax=librosa.note_to_hz('C7'),sr=SR,frame_length=4096,hop_length=256,fill_na=np.nan)
    t=np.arange(len(f0))*256/SR;f=np.where(v&np.isfinite(f0),f0,0.);return t,f


def hz_pesto(x):
    import torch,pesto
    t,p,c,_=pesto.predict(torch.from_numpy(x.astype(np.float32)),SR);t,p,c=[np.asarray(a.detach().cpu() if hasattr(a,'detach') else a,dtype=float) for a in(t,p,c)]
    t=t/(1000. if t.max()>SEC*2 else 1.);return t,np.where((c>.5)&(p>0),p,0.)


def hz_praat(x):
    import parselmouth
    pr=parselmouth.Sound(x,SR).to_pitch_ac(time_step=.01,pitch_floor=75,pitch_ceiling=800);return pr.xs(),pr.selected_array['frequency']


EXT=dict(pyin=hz_pyin,pesto=hz_pesto,praat=hz_praat)


def labels(t,f,n):
    import librosa
    lab=np.full(n,-1,int);idx=np.clip(np.round(np.asarray(t)/STEP).astype(int),0,n-1);ok=f>0
    for i,h in zip(idx[ok],f[ok]):lab[i]=int(np.rint(librosa.hz_to_midi(h)))
    return lab


def rpa(t,f,rt,rf):
    v=rf>0
    if not v.any():return None
    j=np.clip(np.searchsorted(t,rt[v]),1,len(t)-1);j=np.where(abs(t[j-1]-rt[v])<abs(t[j]-rt[v]),j-1,j)
    ff=f[j];ok=ff>0
    cents=np.where(ok,1200*np.abs(np.log2(np.where(ok,ff,1.)/rf[v])),1e9)
    return float(np.mean(cents<=50))


def main():
    if sys.argv[1:]==['declare']:
        run.write(OUT/'declaration.json',dict(doc=__doc__,tool_sha256=run.sha(__file__)));print('DECLARED');return
    assert run.read(OUT/'declaration.json')['tool_sha256']==run.sha(__file__)
    meta={int(r['track_id']):r['singer_id'] for r in csv.DictReader(open(VOC/'vocadito_metadata.csv'))}
    noises={n:load(run.WS/'datasets/environmental_noise'/n) for n in NOISE};recs=[]
    for k,i in enumerate(IDS):
        x=load(VOC/'Audio'/f'vocadito_{i}.wav')[:SEC*SR];n=int(len(x)/SR/STEP)
        ref=np.array([[float(a) for a in r] for r in csv.reader(open(VOC/'Annotations/F0'/f'vocadito_{i}_f0.csv'))]);keep=ref[:,0]<len(x)/SR;rt,rf=ref[keep,0],ref[keep,1]
        variants=[('clean',0,x)]
        for nn in NOISE:
            nz=np.resize(noises[nn][SR:],len(x)) if len(noises[nn])>len(x)+SR else np.resize(noises[nn],len(x))
            r0=np.sqrt(np.mean(x**2))+1e-12;nz=nz/(np.sqrt(np.mean(nz**2))+1e-12)
            for snr in(10,0):variants.append((nn.split('-')[-1].split('.')[0]+'|'+nn,snr,x+nz*r0/10**(snr/20)))
        for name,snr,y in variants:
            R=restore(y);lab={};sc={}
            for nm,sig in(('C',y),('R',R)):
                for e,f in EXT.items():
                    t,h=f(sig.astype(np.float32) if e!='praat' else sig.astype(np.float64));lab[(nm,e)]=labels(t,h,n);sc[(nm,e)]=rpa(t,h,rt,rf)
            cC=consensus([lab[('C',e)] for e in EXT]);cR=consensus([lab[('R',e)] for e in EXT]);acc=bool(cR>=cC+.02)
            for e in EXT:
                if sc[('C',e)] is None:continue
                recs.append(dict(group=meta[i],track=i,cond=name,snr=snr,ext=e,C=sc[('C',e)],R=sc[('R',e)],accepted=acc))
        print('progress',k+1,flush=True)
    for r in recs:r['G']=r['R'] if r['accepted'] else r['C']
    out={};hits0=0;harm_ok=True
    for e in EXT:
        clean=[r for r in recs if r['ext']==e and r['snr']==0 and r['cond']=='clean']
        for snr in(10,0):
            R_=[r for r in recs if r['ext']==e and r['snr']==snr and r['cond']!='clean'];g=sorted({r['group'] for r in R_})
            gm=lambda f:np.array([np.mean([f(r) for r in R_ if r['group']==x]) for x in g])
            gain=gm(lambda r:r['G']-r['C']);boot=gain[np.random.default_rng(339).integers(0,len(g),size=(4000,len(g)))].mean(axis=1);lo=float(np.quantile(boot,.025))
            cl=float(np.mean([np.mean([c['C'] for c in clean if c['group']==x]) for x in g if any(c['group']==x for c in clean)]))
            hg=float(np.mean([r['G']<r['C']-.02 for r in R_]));harm_ok&=hg<=.05
            if snr==0 and lo>0:hits0+=1
            mC,mR,mG=[float(gm(lambda r,k=k:r[k]).mean()) for k in('C','R','G')]
            out[f'{e}|{snr}dB']=dict(clean_ceiling=cl,macro_C=mC,macro_R=mR,macro_G=mG,gain_G=float(gain.mean()),gain_ci_lo=lo,loss_recovered_G=(mG-mC)/(cl-mC) if cl-mC>1e-9 else None,
                harm_R=float(np.mean([r['R']<r['C']-.02 for r in R_])),harm_G=hg,accept_rate=float(np.mean([r['accepted'] for r in R_])),groups=len(g))
    res=dict(summary=out,harm_ok=bool(harm_ok),zero_db_trackers_with_gain=int(hits0),success=bool(harm_ok and hits0>=2))
    run.write(OUT/'results.json',dict(res,records=recs,declaration_sha256=run.sha(OUT/'declaration.json'),no_fit=True))
    print('REAL',json.dumps(res),flush=True)


main()
