"""Damage map: which recording corruptions hurt which frozen pitch/note extractors? No fit, no restoration. Synthetic corruptions of the 53 clean working windows (first 6 s).
Extractors (frozen, defaults): (1) team pYIN baseline params (librosa.pyin, frame 4096, hop 256, fmin C1, fmax C7), (2) PESTO 2.0.1 (confidence>0.5 voiced), (3) Praat autocorrelation (parselmouth, floor 75, ceil 800, step 10 ms).
Label per 20 ms frame = nearest MIDI, unvoiced=-1. For each extractor the REFERENCE is its own output on the clean window (degradation measure, NOT absolute accuracy; no human labels).
Metrics per clip: label_recall = share of clean-voiced frames whose corrupted label equals the clean label (lost voicing counts as error); extra_voiced = share of clean-unvoiced frames labelled voiced. Singer-macro.
Conditions: white noise SNR 10,0 dB; pink noise SNR 10,0; hard clipping at 0.3,0.1 of peak; gain -60,-70 dB with 16-bit quantization; synthetic exponential RIR RT60 0.4,0.8 s;
telephone bandpass 300-3400 Hz; lowpass 1.5 kHz; plus 'clean' control (must give 1.0). Noise/RIR are synthetic stand-ins (no MUSAN/DEMAND/real RIRs used).
Rule (declared before scores): a restoration operator for corruption family F is worth building iff the singer-macro label_recall drop vs clean is >=0.05 for >=2 of 3 extractors at some level of F; otherwise skip F. Report all numbers; no thresholds change after scores."""
import sys,json
import numpy as np
import soundfile as sf
from scipy import signal
import compare_tail_preservation as run
OUT=run.WS/'results/transformer_usability/2026-11-05_damage_map'
SR=22050;STEP=.02;SEC=6
def pink(n,rng):
    X=np.fft.rfft(rng.standard_normal(n));f=np.arange(len(X));f[0]=1;X=X/np.sqrt(f);y=np.fft.irfft(X,n);return y/np.std(y)
def corrupt(x,kind,v,rng):
    r=np.sqrt(np.mean(x**2))+1e-12
    if kind in('white','pink'):
        n=rng.standard_normal(len(x)) if kind=='white' else pink(len(x),rng);return x+n*r/10**(v/20)
    if kind=='clip':p=np.max(abs(x))*v;return np.clip(x,-p,p)
    if kind=='gain':y=x*10**(-abs(v)/20);return np.round(y*32767)/32767
    if kind=='reverb':
        t=np.arange(int(v*SR))/SR;h=rng.standard_normal(len(t))*10**(-3*t/v);h[0]=np.sqrt(np.sum(h[1:]**2))*0.7;y=signal.fftconvolve(x,h)[:len(x)];return y*r/(np.sqrt(np.mean(y**2))+1e-12)
    if kind=='bandpass':sos=signal.butter(4,[300,3400],'bandpass',fs=SR,output='sos');return signal.sosfilt(sos,x)
    if kind=='lowpass':sos=signal.butter(4,1500,'lowpass',fs=SR,output='sos');return signal.sosfilt(sos,x)
    return x
CONDS=[('clean',0),('white',10),('white',0),('pink',10),('pink',0),('clip',.3),('clip',.1),('gain',-60),('gain',-70),('reverb',.4),('reverb',.8),('bandpass',0),('lowpass',0)]
def grid(t,midi,n):
    lab=np.full(n,-1,int);idx=np.clip(np.round(np.asarray(t)/STEP).astype(int),0,n-1)
    for i,m in zip(idx,midi):lab[i]=m
    return lab
def ex_pyin(x,n):
    import librosa
    f0,v,_=librosa.pyin(x,fmin=librosa.note_to_hz('C1'),fmax=librosa.note_to_hz('C7'),sr=SR,frame_length=4096,hop_length=256,fill_na=np.nan)
    t=np.arange(len(f0))*256/SR;ok=v&np.isfinite(f0)&(f0>0);m=np.full(len(f0),-1);m[ok]=np.rint(librosa.hz_to_midi(f0[ok])).astype(int);return grid(t,m,n)
_pesto=None
def ex_pesto(x,n):
    import torch,pesto,librosa
    t,p,c,_=pesto.predict(torch.from_numpy(x.astype(np.float32)),SR);t,p,c=[np.asarray(a.detach().cpu() if hasattr(a,'detach') else a,dtype=float) for a in(t,p,c)];t=t/(1000. if t.max()>SEC*2 else 1.)
    ok=(c>.5)&(p>0);m=np.full(len(p),-1);m[ok]=np.rint(librosa.hz_to_midi(p[ok])).astype(int);return grid(t,m,n)
def ex_praat(x,n):
    import parselmouth,librosa
    pr=parselmouth.Sound(x.astype(np.float64),SR).to_pitch_ac(time_step=.01,pitch_floor=75,pitch_ceiling=800);f=pr.selected_array['frequency'];t=pr.xs()
    ok=f>0;m=np.full(len(f),-1);m[ok]=np.rint(librosa.hz_to_midi(f[ok])).astype(int);return grid(t,m,n)
EXT=dict(pyin=ex_pyin,pesto=ex_pesto,praat=ex_praat)
def main():
    if sys.argv[1:]==['declare']:
        run.write(OUT/'declaration.json',dict(doc=__doc__,tool_sha256=run.sha(__file__)));print('DECLARED');return
    assert run.read(OUT/'declaration.json')['tool_sha256']==run.sha(__file__)
    d,rows,originals=run.declare();rng=np.random.default_rng(339);recs=[]
    for k,o in enumerate([o for o in originals if 'excluded' not in o]):
        x,sr=sf.read(o['path_A'],dtype='float32');assert sr==SR;x=x[:SEC*SR];n=int(len(x)/SR/STEP)
        clean={e:f(x,n) for e,f in EXT.items()}
        for kind,v in CONDS:
            y=corrupt(x.astype(np.float64),kind,v,rng).astype(np.float32)
            for e,f in EXT.items():
                lab=clean[e] if kind=='clean' else f(y,n);cv=clean[e]!=-1;cu=~cv
                recs.append(dict(group=o['group'],src=o['source_id'],kind=kind,level=v,ext=e,recall=float(np.mean(lab[cv]==clean[e][cv])) if cv.any() else None,
                    extra=float(np.mean(lab[cu]!=-1)) if cu.any() else None))
        print('progress',k+1,flush=True)
    res={}
    for e in EXT:
        for kind,v in CONDS:
            R=[r for r in recs if r['ext']==e and r['kind']==kind and r['level']==v and r['recall'] is not None];g=sorted({r['group'] for r in R})
            res[f'{e}|{kind}|{v}']=dict(recall=float(np.mean([np.mean([r['recall'] for r in R if r['group']==x]) for x in g])),
                extra=float(np.mean([np.mean([r['extra'] for r in R if r['group']==x and r['extra'] is not None] or [0]) for x in g])),clips=len(R),groups=len(g))
    fam={}
    for kind in ('white','pink','clip','gain','reverb','bandpass','lowpass'):
        ls=[v for k,v in CONDS if k==kind];hits=0
        for e in EXT:
            drops=[res[f'{e}|clean|0']['recall']-res[f'{e}|{kind}|{v}']['recall'] for v in ls];hits+=max(drops)>=.05
        fam[kind]=dict(extractors_with_drop_ge_0p05=int(hits),build_operator=bool(hits>=2))
    run.write(OUT/'results.json',dict(table=res,families=fam,declaration_sha256=run.sha(OUT/'declaration.json'),no_fit=True,synthetic=True))
    print('DAMAGE',json.dumps(fam),flush=True)
main()
