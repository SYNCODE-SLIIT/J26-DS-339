"""Actual local deviations with independently observed context on both sides."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import sys
import numpy as np
import soundfile as sf
WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.repair.held_voice_policy import eligibility,spans
from bpdr_input.repair.framewise_renderer import render_candidate
from bpdr_input.repair.interventions import audit_shift
from bpdr_input.repair.rule_correction import fine_hz
ROOT=WORKSPACE/'results/transformer_usability/2026-10-28_v5'


def worker(task):
    import torch
    torch.set_num_threads(2)
    if len(task)==3:index,source,spec=task
    else:index,source=task;spec={}
    root=Path(spec.get('root',ROOT));minimum=float(spec.get('minimum_width',1.24));maximum=float(spec.get('maximum_width',1.52))
    context=float(spec.get('context_seconds',.52));fraction=float(spec.get('region_fraction',.5));trim=int(spec.get('trim_frames',6))
    receipt=root/f'source_{index:02d}.json'
    if receipt.exists():return ex.read(receipt)
    pairs=[]
    if 'working_audio' not in source:
        result=dict(source=source,pairs=[dict(source=source['source'],cents=c,excluded='NO_WORKING_ORIGINAL') for c in (-30.,-20.,20.,30.)]);ex.write(receipt,result);return result
    path=Path(source['working_audio']);audio,before=recovery.from_cached(str(path),path.parent/'original_features.npz',{})
    times=before['features']['times_seconds'];raw=before['views'][0]
    allowed,reliable,_=eligibility(raw,region_split_cents=70.)
    eligible=[]
    for start,stop in spans(allowed&reliable):
        a,b=start+trim,stop-trim
        if (b-a)*.02<max(minimum+2*context,minimum/fraction):continue
        y=raw[a:b,64]
        if spec.get('quality_median_frames'):
            from scipy.ndimage import median_filter
            y=median_filter(y,size=spec['quality_median_frames'],mode='nearest')
        x=np.arange(len(y));spread=float(np.std(y-np.polyval(np.polyfit(x,y,1),x)))
        if spread<=10.:eligible.append((a,b,spread))
    folder=root/'prepared'/f'{index:02d}';folder.mkdir(parents=True,exist_ok=True)
    baseline=folder/'original_working.wav';baseline.write_bytes(path.read_bytes())
    (folder/'original_features.npz').write_bytes((path.parent/'original_features.npz').read_bytes())
    for cents in (-30.,-20.,20.,30.):
        base=dict(source=source['source'],group=source['source']['singer_id'],source_id=Path(source['source']['path']).name,cents=cents,path_A=str(baseline))
        if not eligible:pairs.append(dict(**base,excluded='NO_STABLE_HELD_REGION_WITH_TWO_SIDED_CONTEXT'));continue
        a,b,spread=max(eligible,key=lambda r:r[1]-r[0])
        width=min(maximum,fraction*(b-a)*.02,.24*len(audio)/22050)
        if width<minimum:pairs.append(dict(**base,excluded='INSUFFICIENT_CONTEXT_OR_DURATION_BUDGET'));continue
        seed=int(hashlib.sha256((source['source']['sha256']+str(cents)+spec.get('version','recoverable-v5')).encode()).hexdigest()[:12],16)
        rng=np.random.default_rng(seed);left=float(rng.uniform(a*.02+context,b*.02-context-width));support=(left,left+width)
        curve=np.where((times>=left)&(times<left+width),cents,0.)
        candidate,_=render_candidate(audio,22050,times,curve,[support]);out=folder/f'detuned_{cents:+g}.wav'
        sf.write(out,candidate,22050,subtype='PCM_24');candidate,_=sf.read(out,dtype='float32')
        after=recovery.measured(candidate)
        audit,mask,target=audit_shift(before['features'],after['features'],fine_hz(audio,22050,times),fine_hz(candidate,22050,times),before['reliable']&after['reliable'],support,cents)
        ex.write(folder/f'audit_{cents:+g}.json',audit)
        if not audit['passed']:pairs.append(dict(**base,excluded='INTERVENTION_AUDIT_FAILED',audit=audit,support_seconds=support));continue
        target_path=folder/f'targets_{cents:+g}_NOT_MODEL_INPUT.npz'
        np.savez_compressed(target_path,target=target.astype(np.float32),mask=mask,times=times)
        np.savez_compressed(folder/f'features_{cents:+g}.npz',**after['features'])
        pairs.append(dict(**base,path_C=str(out),target_path=str(target_path),support_seconds=support,audit=audit))
    result=dict(source=source,pairs=pairs);ex.write(receipt,result);return result


if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    sources=ex.read(ROOT.parent/'2026-10-28_long_v2/data_declaration.json')['source_records']
    if not (ROOT/'data_declaration.json').exists():ex.write(ROOT/'data_declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        reason='Training-only feature ceiling retained originals but could not identify many mostly-detuned short notes. Declare genuinely local deviations with >=.52s measured original context on BOTH sides; never pad a runtime repair.',
        source_records=sources,scope='Stable held acoustic regions, interior detrended spread<=10c, local actual deviation1.24-1.52s and <=50% held region, <=24% full audio duration',
        eligibility='Version2, acoustic pitch-region split70c; original controller and delivered audits unchanged',feature='Input-only plateau-plus-linear-drift evidence; not a reference melody or known target mask',
        final_singers_not_accessed=['female8','female9','male8','male9'],source_sha256=ex.sha(__file__)))
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i,result in enumerate(pool.map(worker,enumerate(sources,1)),1):
            rows.extend(result['pairs']);ex.write(ROOT/'preparation_progress.json',dict(completed_sources=i,planned_sources=len(sources),attempted=len(rows),scoreable=sum('excluded' not in r for r in rows)))
            print('RECOVERABLE_STEP_SOURCE',i,'/',len(sources),'valid',sum('excluded' not in r for r in rows),'/',len(rows),flush=True)
    ex.write(ROOT/'train_pairs.json',rows);print('RECOVERABLE_STEP_TRAINING_READY',flush=True)
