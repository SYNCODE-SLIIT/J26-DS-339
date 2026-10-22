"""Bounded source-preservation study: real training recordings + known added noise.

No fitting, no final-test access, no claim that source recordings are noise-free.
The clean reference below means baseline source before the *added* noise only.
"""
from bpdr_input import module_source
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from bpdr_input.repair.noise_preview import NoisePolicy, attenuate, audit, COMPONENT
from bpdr_input.experiments.evaluation.transfer_study import read, sha, write, FINE


def main():
    import torch
    from scipy.signal import resample_poly, lfilter
    from math import gcd
    from bpdr_input.audio.analysis import extract
    torch.set_num_threads(2)
    out=COMPONENT/'runs/noise_study'/uuid.uuid4().hex;out.mkdir(parents=True)
    records=read(FINE/'training_catalog.json')['records']
    selected=[]
    for dataset in ['vocadito','jamendo']:
        by_phrase={r['phrase_id']:r for r in records if r['dataset']==dataset}
        selected += [dict(dataset=dataset,phrase_id=k,source=by_phrase[k]['waveform_paths']['A'],
                          split='train',group=by_phrase[k].get('group_id',by_phrase[k]['singer_id']))
                     for k in sorted(by_phrase)[:3]]
    root=Path(read(COMPONENT/'runs/controlled_study/current.json')['path'])
    by_voice={}
    for r in sorted(read(root/'catalog.json')['records'],key=lambda r:r['path']):
        if r['split']=='train':
            p=Path(r['path']).parent/'A.wav'
            by_voice.setdefault(p.parents[2].name,p)
    selected += [dict(dataset='synthetic',phrase_id=Path(p).parents[1].name,source=str(p),split='train',group=voice)
                 for voice,p in sorted(by_voice.items())[:2]]
    # Freeze all choices, thresholds and source hashes before any result.
    protocol=dict(version='noise-study-1',seed=1701,policy=asdict(NoisePolicy()),sources=selected,
        source_sha256={s['source']:sha(s['source']) for s in selected},
        implementation_sha256={str(p):sha(p) for p in [Path(__file__),module_source('noise_preview.py')]},
        snr_db=[10,20],noise_types=['white','broadband_fan_proxy'],prefix_seconds=.8,
        source_excerpt_seconds=8.,noise_reference=[0.,.7],
        interpretation='Numerical preservation and removal of known synthetic noise from baseline audio; not natural-noise or novice-user validation',
        additional_score_gate='Positive SNR gain against baseline source, over all samples and active-source samples',
        model_training=False,final_test_access=False,prepared_handoff_enabled=False)
    write(out/'protocol.json',protocol)
    rng=np.random.default_rng(1701);rows=[]
    for i,s in enumerate(selected):
        x,sr=sf.read(s['source'],dtype='float64')
        if x.ndim==2:x=x[:,0]
        g=gcd(sr,22050);x=resample_poly(x,22050//g,sr//g) if sr!=22050 else x
        sr=22050;x=x[:8*sr];baseline=np.r_[np.zeros(round(.8*sr)),x]
        baseline*=.6/max(.6,np.abs(baseline).max())
        active=np.abs(baseline)>max(1e-4,.02*np.abs(baseline).max())
        level=np.sqrt(np.mean(baseline[active]**2))
        for kind in protocol['noise_types']:
            noise=rng.normal(size=len(baseline))
            if kind=='broadband_fan_proxy':
                colored=lfilter([1.],[1.,-.9],noise)
                noise=noise+.5*colored/np.std(colored)
            noise/=np.sqrt(np.mean(noise**2))
            for snr in protocol['snr_db']:
                folder=out/f'{i:02d}_{kind}_{snr}';folder.mkdir()
                mixed=baseline+noise*level*10**(-snr/20)
                # Avoid clipping by scaling all three signals equally.
                scale=.95/max(.95,np.abs(mixed).max());clean=baseline*scale;mixed*=scale
                candidate,receipt=attenuate(mixed,sr,(0.,.7))
                before,_=extract(mixed,sr,COMPONENT/'checkpoints')
                # Score actual encoded candidate, not an unencoded array.
                sf.write(folder/'baseline.wav',clean,sr,subtype='PCM_24')
                sf.write(folder/'noisy.wav',mixed,sr,subtype='PCM_24')
                sf.write(folder/'candidate.wav',candidate,sr,subtype='PCM_24')
                candidate,_=sf.read(folder/'candidate.wav')
                after,_=extract(candidate,sr,COMPONENT/'checkpoints')
                check=audit(mixed,candidate,sr,before,after,(0.,.7))
                def gain(mask):
                    return float(10*np.log10((np.mean((mixed[mask]-clean[mask])**2)+1e-20)/
                                             (np.mean((candidate[mask]-clean[mask])**2)+1e-20)))
                total_gain=gain(np.ones(len(clean),bool));active_gain=gain(active)
                row=dict(source=s,noise=kind,nominal_snr_db=snr,directory=str(folder),receipt=receipt,audit=check,
                    snr_gain_db=total_gain,active_source_snr_gain_db=active_gain,
                    passes=bool(receipt['outcome']=='UNVERIFIED_NOISE_CANDIDATE' and check['passes'] and total_gain>0 and active_gain>0))
                write(folder/'result.json',row);rows.append(row)
                print(s['dataset'],s['phrase_id'],kind,snr,'pass',row['passes'],'SNR gain',round(total_gain,3),'reasons',check['reasons'],flush=True)
    for p,h in protocol['source_sha256'].items():
        if sha(p)!=h:raise ValueError('Source modified')
    for p,h in protocol['implementation_sha256'].items():
        if sha(p)!=h:raise ValueError('Study implementation modified during execution')
    summary={d:dict(cases=len(rr),passes=sum(r['passes'] for r in rr),
                median_snr_gain_db=float(np.median([r['snr_gain_db'] for r in rr])),
                median_active_source_snr_gain_db=float(np.median([r['active_source_snr_gain_db'] for r in rr])))
             for d in ['vocadito','jamendo','synthetic'] if (rr:=[r for r in rows if r['source']['dataset']==d])}
    write(out/'evidence.json',dict(protocol=protocol,rows=rows,summary=summary,
        limitations=['Small train-only engineering sample','Added synthetic noise; no naturally noisy paired ground truth',
                     'No listener study or musical/intended-note judgement','Thresholds provisional; not development calibrated'],
        original_sources_unchanged=True))
    write(out.parent/'current.json',dict(path=str(out)))
    print(json.dumps(summary,indent=2));print('RUN',out)


if __name__=='__main__':main()
