"""Neural denoiser arm of the real-noise test. Same 21 training-role Vocadito recordings (first 8 s), same ESC-50 root clips at 10 and 0 dB, same trackers/metric (RPA50 vs HUMAN F0) and same input-only
3-tracker consensus gate (accept if consensus rises >= 0.02) as real_noise_human_f0.py. No fit.
Modes: 'mix' writes the identical noisy variants as 22.05 kHz WAVs; an isolated environment (DeepFilterNet 0.5.6 + torch 2.2.2, NOT the project environment) writes enhanced WAVs; 'score' evaluates.
Neural arm N = level normalisation (same detector) -> DeepFilterNet3 -> gate. Arms compared: C noisy, S spectral-gate R (round 1 pipeline, recomputed here), N neural, GN = gated N, GS = gated S.
Rule (declared before scores): the neural arm is useful iff at 0 dB (i) singer-macro RPA gain GN-C has a bootstrap 95% lower bound > 0 for ALL 3 trackers, (ii) share of noise-induced loss recovered (GN-C)/(clean-C) >= 0.25 for >= 2 trackers,
(iii) gated harm (per-recording RPA drop > 0.02 vs C) <= 5% for every tracker. Also report N ungated, GS, and whether GN beats GS.
Limits: 4 noise clips, 21 recordings; consensus proxy shares trackers with scoring; DeepFilterNet trained on speech (singing/humming out of domain); human F0 is performed pitch; not independently replayed."""
import sys,json
import numpy as np
import soundfile as sf
import compare_tail_preservation as run
ns={}
exec(open(run.WS/'tools/real_noise_human_f0.py',encoding='utf8').read().rsplit('\n\nmain()',1)[0],ns)
OUT=run.WS/'results/transformer_usability/2026-11-05_real_noise_neural'
SR,SEC,STEP=ns['SR'],ns['SEC'],ns['STEP'];IDS=ns['IDS'];NOISE=ns['NOISE'];VOC=ns['VOC'];EXT=ns['EXT'];load=ns['load'];labels=ns['labels'];rpa=ns['rpa']
restore,consensus=ns['restore'],ns['consensus'];level=ns['g1']['level']
import csv


def variants(i):
    x=load(VOC/'Audio'/f'vocadito_{i}.wav')[:SEC*SR];out=[('clean',0,x)]
    for nn in NOISE:
        nz=load(run.WS/'datasets/environmental_noise'/nn);nz=np.resize(nz,len(x));r0=np.sqrt(np.mean(x**2))+1e-12;nz=nz/(np.sqrt(np.mean(nz**2))+1e-12)
        for snr in(10,0):out.append((nn.split('-')[-1].split('.')[0],snr,x+nz*r0/10**(snr/20)))
    return x,out


def main():
    mode=sys.argv[1] if len(sys.argv)>1 else ''
    if mode=='declare':
        run.write(OUT/'declaration.json',dict(doc=__doc__,tool_sha256=run.sha(__file__)));print('DECLARED');return
    assert run.read(OUT/'declaration.json')['tool_sha256']==run.sha(__file__)
    mix=OUT/'mix';enh=OUT/'enh'
    if mode=='mix':
        mix.mkdir(exist_ok=True)
        for i in IDS:
            x,vs=variants(i)
            for name,snr,y in vs:
                sf.write(mix/f'{i}_{name}_{snr}.wav',level(y).astype(np.float32),SR,subtype='FLOAT')
        print('MIXED',len(list(mix.glob('*.wav'))));return
    meta={int(r['track_id']):r['singer_id'] for r in csv.DictReader(open(VOC/'vocadito_metadata.csv'))};recs=[]
    for k,i in enumerate(IDS):
        x,vs=variants(i);n=int(len(x)/SR/STEP)
        ref=np.array([[float(a) for a in r] for r in csv.reader(open(VOC/'Annotations/F0'/f'vocadito_{i}_f0.csv'))]);keep=ref[:,0]<len(x)/SR;rt,rf=ref[keep,0],ref[keep,1]
        for name,snr,y in vs:
            N,_=sf.read(enh/f'{i}_{name}_{snr}.wav',dtype='float64');N=np.resize(N,len(y));S=restore(y);lab={};sc={}
            for nm,sig in(('C',y),('S',S),('N',N)):
                for e,f in EXT.items():
                    t,h=f(sig.astype(np.float32) if e!='praat' else sig.astype(np.float64));lab[(nm,e)]=labels(t,h,n);sc[(nm,e)]=rpa(t,h,rt,rf)
            cC=consensus([lab[('C',e)] for e in EXT]);cS=consensus([lab[('S',e)] for e in EXT]);cN=consensus([lab[('N',e)] for e in EXT])
            for e in EXT:
                if sc[('C',e)] is None:continue
                recs.append(dict(group=meta[i],track=i,cond=name,snr=snr,ext=e,C=sc[('C',e)],S=sc[('S',e)],N=sc[('N',e)],accS=bool(cS>=cC+.02),accN=bool(cN>=cC+.02)))
        print('progress',k+1,flush=True)
    for r in recs:r['GS']=r['S'] if r['accS'] else r['C'];r['GN']=r['N'] if r['accN'] else r['C']
    out={};ok_lo=0;ok_rec=0;ok_h=True
    for e in EXT:
        clean=[r for r in recs if r['ext']==e and r['cond']=='clean']
        for snr in(10,0):
            R_=[r for r in recs if r['ext']==e and r['snr']==snr and r['cond']!='clean'];g=sorted({r['group'] for r in R_})
            gm=lambda f:np.array([np.mean([f(r) for r in R_ if r['group']==x]) for x in g])
            gain=gm(lambda r:r['GN']-r['C']);boot=gain[np.random.default_rng(339).integers(0,len(g),size=(4000,len(g)))].mean(axis=1);lo=float(np.quantile(boot,.025))
            cl=float(np.mean([np.mean([c['C'] for c in clean if c['group']==x]) for x in g if any(c['group']==x for c in clean)]))
            m={k:float(gm(lambda r,k=k:r[k]).mean()) for k in('C','S','N','GS','GN')};rec=(m['GN']-m['C'])/(cl-m['C']) if cl-m['C']>1e-9 else None
            hg=float(np.mean([r['GN']<r['C']-.02 for r in R_]))
            if snr==0:
                ok_lo+=bool(lo>0);ok_rec+=bool(rec is not None and rec>=.25);ok_h&=hg<=.05
            out[f'{e}|{snr}dB']=dict(clean_ceiling=cl,**m,gain_GN=float(gain.mean()),gain_GN_ci_lo=lo,loss_recovered_GN=rec,loss_recovered_GS=(m['GS']-m['C'])/(cl-m['C']) if cl-m['C']>1e-9 else None,
                harm_N=float(np.mean([r['N']<r['C']-.02 for r in R_])),harm_GN=hg,accept_N=float(np.mean([r['accN'] for r in R_])),accept_S=float(np.mean([r['accS'] for r in R_])))
    res=dict(summary=out,zero_db_trackers_ci_pos=int(ok_lo),zero_db_trackers_recovery_ge_25=int(ok_rec),harm_ok=bool(ok_h),neural_useful=bool(ok_lo==3 and ok_rec>=2 and ok_h))
    run.write(OUT/'results.json',dict(res,records=recs,declaration_sha256=run.sha(OUT/'declaration.json'),no_fit=True))
    print('NEURAL',json.dumps(res),flush=True)


main()
