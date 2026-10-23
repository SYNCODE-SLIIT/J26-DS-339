"""Fixed-policy stress test with field-recorded environmental noise.

Mixtures have a known baseline, not naturally co-recorded novice/clean pairs.
Breathing is never an eligible noise reference and is tested as abstention.
"""
from pathlib import Path
from math import gcd
import uuid
import json
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from bpdr_input.repair.noise_preview import NoisePolicy, attenuate, audit
from bpdr_input.experiments.evaluation.transfer_study import COMPONENT, read, write, sha


def main(policy=None,noise_receipt=None):
    import torch
    from dataclasses import asdict
    from bpdr_input.audio.analysis import extract
    torch.set_num_threads(2)
    policy=policy or NoisePolicy()
    out=COMPONENT/'runs/environmental_noise_study'/uuid.uuid4().hex;out.mkdir(parents=True)
    previous=Path(read(COMPONENT/'runs/noise_study/current.json')['path'])
    sources=[]
    for ds in ['vocadito','jamendo']:
        sources += [s for s in read(previous/'protocol.json')['sources'] if s['dataset']==ds][:2]
    intake=read(COMPONENT/'data/humtrans/intake_receipt.json')['records'];seen=set()
    for r in intake:
        if r['split']=='train' and r['person_id'] not in seen:
            sources.append(dict(dataset='humtrans',source=r['path'],phrase_id=r['key'],group=r['person_id'],split='train'));seen.add(r['person_id'])
        if len(seen)==2:break
    receipt_path=Path(noise_receipt) if noise_receipt else COMPONENT/'data/environmental_noise/receipt.json'
    noises=read(receipt_path)['records']
    declaration=dict(version='field-noise-mixtures-1',source_sha256=sha(__file__),policy=asdict(policy),noise_receipt_sha256=sha(receipt_path),sources=sources,noises=noises,
        source_hashes={s['source']:sha(s['source']) for s in sources},snr_db=[10.,20.],prefix_seconds=.8,reference_interval=[0.,.7],excerpt_seconds=8.,
        mixing='Field-recorded five-second noises resampled and repeated; scaled relative to baseline active-sample RMS; common anti-clipping gain',
        breathing='Vocal sound, never declared noise-only; expected exact unchanged abstention',
        evaluation='All cases; numerical preservation plus positive overall and active-source SNR gain against baseline',
        interpretation='Additional recorded-noise augmentation, not naturally co-recorded humming/clean pairs or a listening study',
        model_training=False,final_test_audio_accessed=False,research_ready_examples=0)
    write(out/'declaration.json',declaration);(out/'executed_environmental_noise_study.py').write_bytes(Path(__file__).read_bytes());rows=[]
    for index,s in enumerate(sources):
        x,sr=sf.read(s['source'],dtype='float64')
        if x.ndim==2:x=x[:,np.argmax(np.mean(x*x,axis=0))]
        x=x[:8*sr];g=gcd(sr,22050);x=resample_poly(x,22050//g,sr//g);sr=22050
        baseline=np.r_[np.zeros(round(.8*sr)),x];baseline*=.6/max(.6,np.abs(baseline).max())
        active=np.abs(baseline)>max(1e-4,.02*np.abs(baseline).max());level=np.sqrt(np.mean(baseline[active]**2))
        for noise in noises:
            n,nsr=sf.read(noise['path']);g=gcd(nsr,sr);n=resample_poly(n,sr//g,nsr//g)
            n=np.resize(n,len(baseline));n/=np.sqrt(np.mean(n*n))
            for snr in ([20.] if noise['category']=='breathing' else declaration['snr_db']):
                folder=out/f'{index}_{noise["category"]}_{snr:g}';folder.mkdir()
                mixed=baseline+n*level*10**(-snr/20.);gain=.95/max(.95,np.abs(mixed).max());clean=baseline*gain;mixed*=gain
                interval=None if noise['category']=='breathing' else (0.,.7)
                candidate,receipt=attenuate(mixed,sr,interval,policy)
                check=None
                if receipt['outcome']=='UNVERIFIED_NOISE_CANDIDATE':
                    before,_=extract(mixed,sr,COMPONENT/'checkpoints');after,_=extract(candidate,sr,COMPONENT/'checkpoints')
                    check=audit(mixed,candidate,sr,before,after,interval,policy)
                def improvement(mask):
                    return float(10*np.log10((np.mean((mixed[mask]-clean[mask])**2)+1e-20)/(np.mean((candidate[mask]-clean[mask])**2)+1e-20)))
                overall=improvement(np.ones(len(clean),bool));voice=improvement(active)
                passed=bool(check and check['passes'] and overall>0 and voice>0)
                unchanged=bool(np.array_equal(mixed,candidate))
                for name,a in [('baseline.wav',clean),('noisy.wav',mixed),('candidate.wav',candidate)]:sf.write(folder/name,a,sr,subtype='PCM_24')
                row=dict(dataset=s['dataset'],phrase_id=s['phrase_id'],noise=noise['category'],snr_db=snr,receipt=receipt,audit=check,
                    snr_gain_db=overall,active_source_snr_gain_db=voice,passes=passed,exact_identity=unchanged,directory=str(folder))
                rows.append(row);write(folder/'result.json',row)
                print(s['dataset'],noise['category'],snr,'pass',passed,'identity',unchanged,flush=True)
    for p,h in declaration['source_hashes'].items():
        if sha(p)!=h:raise ValueError('Source changed')
    summary={}
    for kind in sorted({r['noise'] for r in rows}):
        rr=[r for r in rows if r['noise']==kind];summary[kind]=dict(cases=len(rr),passes=sum(r['passes'] for r in rr),unchanged=sum(r['exact_identity'] for r in rr),
            median_snr_gain_db=float(np.median([r['snr_gain_db'] for r in rr])))
    write(out/'evidence.json',dict(declaration_sha256=sha(out/'declaration.json'),summary=summary,rows=rows,
        original_sources_unchanged=True,prepared_handoff_enabled=False,
        limitations=['Recorded-noise mixtures, not naturally noisy participant recordings','Numerical tracker checks, no listening approval','Small correlated subset and provisional fixed policy']))
    pointer='colored_reference_current.json' if policy.version=='declared-colored-reference-preview-2' else 'original_policy_current.json'
    write(out.parent/pointer,dict(path=str(out)))
    if pointer=='colored_reference_current.json' or not (out.parent/'current.json').exists():
        write(out.parent/'current.json',dict(path=str(out)))
    print('NOISE_STRESS_COMPLETE',json.dumps(summary),out,flush=True)


if __name__=='__main__':
    import sys
    if len(sys.argv)>1:
        from bpdr_input.repair.noise_policy_v2 import ColoredNoisePolicy
        main(ColoredNoisePolicy(),sys.argv[1])
    else:main()
