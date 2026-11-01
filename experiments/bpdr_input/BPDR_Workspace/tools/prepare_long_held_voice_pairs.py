"""Create a separately declared longer-deviation training condition, real WAVs.

No runtime candidate is padded. These are longer actual injected deviations,
audited independently before labels are used. Final singers remain untouched.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import sys
import numpy as np
import soundfile as sf

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as existing
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.repair.held_voice_policy import eligibility,spans
from bpdr_input.repair.framewise_renderer import render_candidate
from bpdr_input.repair.interventions import audit_shift
from bpdr_input.repair.rule_correction import fine_hz

ROOT=WORKSPACE/'results/transformer_usability/2026-10-28_long_v2'


def worker(task):
    import torch
    torch.set_num_threads(2)
    i,source=task;receipt=ROOT/f'source_{i:02d}.json'
    if receipt.exists():return existing.read(receipt)
    if 'working_audio' not in source:
        pairs=[dict(source=source['source'],cents=c,excluded=source.get('excluded','No working source')) for c in (-30.,-20.,20.,30.)]
        result=dict(source=source,pairs=pairs);existing.write(receipt,result);return result
    original=Path(source['working_audio'])
    audio,before=recovery.from_cached(str(original),original.parent/'original_features.npz',{})
    times=before['features']['times_seconds'];allowed,reliable,_=eligibility(before['views'][0])
    eligible=[(a,b) for a,b in spans(allowed&reliable) if (b-a)*.02>=1.92]
    pairs=[];folder=ROOT/'prepared'/f'{i:02d}';folder.mkdir(parents=True,exist_ok=True)
    baseline=folder/'original_working.wav'
    baseline.write_bytes(original.read_bytes())
    (folder/'original_features.npz').write_bytes((original.parent/'original_features.npz').read_bytes())
    for cents in (-30.,-20.,20.,30.):
        base=dict(source=source['source'],cents=cents,group=source['source']['singer_id'],source_id=Path(source['source']['path']).name,path_A=str(baseline))
        if not eligible:
            pairs.append(dict(**base,excluded='NO_RELIABLE_LONG_RUN_FOR_LONGER_DEVIATION'));continue
        a,b=max(eligible,key=lambda x:x[1]-x[0])
        width=min(1.84,(b-a)*.02-.40,len(audio)/22050*.24)
        if width<1.52:
            pairs.append(dict(**base,excluded='LONGER_DEVIATION_EXCEEDS_CONTEXT_OR_DURATION_BUDGET'));continue
        seed=int(hashlib.sha256((source['source']['sha256']+str(cents)+'long-v2').encode()).hexdigest()[:12],16)
        rng=np.random.default_rng(seed);left=float(rng.uniform(a*.02+.20,max(a*.02+.20,b*.02-.20-width)))
        support=(left,left+width);curve=np.where((times>=left)&(times<left+width),cents,0.)
        candidate,_=render_candidate(audio,22050,times,curve,[support])
        path=folder/f'detuned_{cents:+g}.wav';sf.write(path,candidate,22050,subtype='PCM_24');candidate,_=sf.read(path,dtype='float32')
        after=recovery.measured(candidate)
        audit,mask,target=audit_shift(before['features'],after['features'],fine_hz(audio,22050,times),fine_hz(candidate,22050,times),before['reliable']&after['reliable'],support,cents)
        existing.write(folder/f'audit_{cents:+g}.json',audit)
        if not audit['passed']:
            pairs.append(dict(**base,excluded='LONG_INTERVENTION_AUDIT_FAILED',support_seconds=support,audit=audit));continue
        target_path=folder/f'targets_{cents:+g}_NOT_MODEL_INPUT.npz'
        np.savez_compressed(target_path,target=target.astype(np.float32),mask=mask,times=times)
        np.savez_compressed(folder/f'features_{cents:+g}.npz',**after['features'])
        pairs.append(dict(**base,path_C=str(path),target_path=str(target_path),support_seconds=support,audit=audit))
    result=dict(source=source,pairs=pairs);existing.write(receipt,result)
    return result


if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    old=Path(existing.read(WORKSPACE/'results/model_recovery/current.json')['path'])
    sources=existing.read(old/'train_sources.json')+[existing.read(p)['source'] for p in sorted((ROOT.parent/'2026-10-28_v1').glob('source_*.json'))]
    if not (ROOT/'data_declaration.json').exists():existing.write(ROOT/'data_declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        reason='V1 training-fold candidate analysis found most learned supports shorter than the unchanged 0.86s render minimum. Evaluate an explicitly longer ACTUAL deviation scope; never extend a runtime candidate to pass permission.',
        scope='Stable held voice, actual deviation width 1.52-1.84s, <=24% source duration; no general short-note repair claim',
        interventions='Praat renderer, four +/-20/30c deviations, source-hash deterministic position, >=.20s original context on both sides; existing independent two-tracker label audit unchanged',
        source_records=sources,final_singers_not_accessed=['female9','male9'],development_singers_not_yet_accessed=['female8','male8'],
        training_sources_only=True,source_sha256=existing.sha(__file__)))
    pairs=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i,record in enumerate(pool.map(worker,enumerate(sources,1)),1):
            pairs.extend(record['pairs']);print('LONG_PREP',i,'/',len(sources),'valid',sum('excluded' not in p for p in pairs),'/',len(pairs),flush=True)
            existing.write(ROOT/'preparation_progress.json',dict(completed_sources=i,planned_sources=len(sources),scoreable=sum('excluded' not in p for p in pairs),attempted=len(pairs)))
    existing.write(ROOT/'train_pairs.json',pairs)
    print('LONG_TRAINING_READY',len(pairs),flush=True)
