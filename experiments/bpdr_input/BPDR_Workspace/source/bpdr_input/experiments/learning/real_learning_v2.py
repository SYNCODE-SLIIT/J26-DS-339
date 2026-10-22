"""Versioned A4-safe real inputs and predeclared balanced/scaled learning checks.

Old data, targets, experiments, model release and shared pipeline stay immutable.
This is exploratory engineering, not a musical reference approval or final test.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import numpy as np
from bpdr_input.experiments.evaluation.transfer_study import COMPONENT, FINE, read, write, sha, score_predictions, summaries

ROOT=COMPONENT/'runs/real_learning_v2'


def relative(raw):
    out=np.array(raw,dtype=np.float32,copy=True)
    voiced=out[:,66]>.5
    out[:,64]=np.where(voiced,out[:,64]-np.median(out[voiced,64]) if voiced.any() else 0.,0.)
    return out


def pair(record):
    from bpdr_input.modeling.examples import read_pair
    p=read_pair(record,'engineering')
    p['inputs']={v:relative(x) for v,x in p['inputs'].items()}
    p['mask'] &= (p['inputs']['A'][:,66]>.5) & (p['inputs']['C'][:,66]>.5)
    if not (p['mask'] & (p['detuning']!=0)).any() or not (p['mask'] & (p['detuning']==0)).any():
        raise ValueError('Pair has no reliable correction/preservation support')
    return p


def hierarchical_weights(records):
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[(r['dataset'],r['singer_id'],r['phrase_id'])].append(i)
    ds=sorted({g[0] for g in groups});weights=np.zeros(len(records))
    for d in ds:
        singers=sorted({g[1] for g in groups if g[0]==d})
        for singer in singers:
            phrases=[g for g in groups if g[:2]==(d,singer)]
            for g in phrases:
                weights[groups[g]]=1/len(ds)/len(singers)/len(phrases)/len(groups[g])
    return weights


def rebuild():
    from bpdr_input.experiments.learning.fine_pitch import yin_cents
    out=ROOT/uuid.uuid4().hex;out.mkdir(parents=True)
    declaration=dict(version='a4-explicit-voicing-2',created_utc=datetime.now(timezone.utc).isoformat(),
        old_run=str(FINE),source_sha256=sha(__file__),
        change='Restore YIN on PESTO-voiced A4-bin frames previously replaced by zero; 50-cent agreement guard; explicit-voicing phrase centering',
        preserve='All other fine pitch values, raw acoustic/boundary columns, source waveforms, existing target labels and old artifacts',
        mask='Existing common mask intersected with corrected A/C voicing at training/evaluation load; no new labels',
        normalizer='Dataset/singer/phrase/variant-equal training-only moments, four views; pitch scale fixed 100 cents, pitch-change scale fixed 30 cents',
        final_test_access=False,review_status='pending',research_ready_examples=0,
        catalogs={s:sha(FINE/(s+'_catalog.json')) for s in ['training','development']})
    write(out/'declaration.json',declaration);print('REBUILD',out,flush=True)
    cache={};counts=defaultdict(lambda:defaultdict(int));catalogs={}
    for split in ['training','development']:
        records=read(FINE/(split+'_catalog.json'))['records'];new=[]
        for i,r in enumerate(records):
            folder=out/'inputs'/split/f'{i:04d}';folder.mkdir(parents=True)
            hashes={}
            for v in 'ABCD':
                old=Path(r['input_directory'])/(v+'_input.npz')
                if sha(old)!=r['input_sha256'][v]:raise ValueError('Old fine input changed')
                with np.load(old) as z:x=z['boundary_features_71'].copy();times=z['times_seconds']
                rawpath=Path(r['quantized_input_directory'])/(v+'_input.npz')
                if sha(rawpath)!=r['quantized_input_sha256'][v]:raise ValueError('Quantized source changed')
                with np.load(rawpath) as z:raw=z['boundary_features_71']
                restore=(raw[:,66]>.5)&(raw[:,64]==0)
                if restore.any():
                    wav=r['waveform_paths'][v]
                    if wav not in cache:
                        if sha(wav)!=r['waveform_sha256'][v]:raise ValueError('Waveform changed')
                        cache[wav]=yin_cents(wav,times)
                    fine=cache[wav];keep=restore & np.isfinite(fine) & (np.abs(fine-raw[:,64])<=50.)
                    x[restore,64]=np.where(keep[restore],fine[restore],0.)
                    x[restore,66]=keep[restore].astype(np.float32)
                    counts[split]['restored_view_frames']+=int(keep.sum())
                    counts[split]['withheld_view_frames']+=int((restore & ~keep).sum())
                voiced=x[:,66]>.5;x[:,67]=0.
                x[1:,67]=np.where(voiced[1:] & voiced[:-1],np.diff(x[:,64]),0.)
                if not np.array_equal(x[:,[j for j in range(71) if j not in [64,66,67]]],raw[:,[j for j in range(71) if j not in [64,66,67]]]):
                    raise ValueError('Unrelated acoustic columns changed')
                p=folder/(v+'_input.npz');np.savez_compressed(p,times_seconds=times,boundary_features_71=x,frame_features_70=x[:,:70]);hashes[v]=sha(p)
            new.append(dict(r,input_directory=str(folder),input_sha256=hashes,pitch_columns='a4-explicit-voicing-2'))
            if (i+1)%100==0:print(split,i+1,'/',len(records),flush=True)
        write(out/(split+'_catalog.json'),dict(records=new));catalogs[split]=new
    weights=hierarchical_weights(catalogs['training']);mean=np.zeros(71);second=mean.copy()
    for r,w in zip(catalogs['training'],weights):
        p=pair(r)
        for x in p['inputs'].values():
            a=x.astype(np.float64);mean+=w/4*a.mean(0);second+=w/4*(a*a).mean(0)
    scale=np.sqrt(np.maximum(second-mean*mean,0));scale[scale<1e-6]=1.
    mean[64]=0.;scale[64]=100.;mean[67]=0.;scale[67]=30.
    np.savez(out/'normalizer.npz',mean=mean.astype(np.float32),scale=scale.astype(np.float32),weights=weights)
    evidence=dict(declaration_sha256=sha(out/'declaration.json'),counts={k:dict(v) for k,v in counts.items()},
        waveforms_reanalysed=len(cache),normalizer_sha256=sha(out/'normalizer.npz'),
        catalogs={s:sha(out/(s+'_catalog.json')) for s in catalogs},source_sha256=sha(__file__),old_sources_unchanged=True)
    write(out/'build_evidence.json',evidence);write(ROOT/'current.json',dict(path=str(out)));print('BUILD_COMPLETE',out,flush=True)


def balanced_objective(pred,batch,condition,response=False):
    """Equal correction/preservation exposure within C/D; all A/B remain zero anchors.

    Cents are scaled by 30 for optimization only. Saved head is converted back
    to physical cents. This anchor differs from the historical frame mean.
    """
    import torch
    from bpdr_input.modeling.losses import smoothness, boundary_invariance, counter_detuning
    m=batch['mask'];det=batch['detuning'];zero=torch.zeros_like(det)
    def loss(p,target,mask):
        n=mask.sum(1);keep=n>0
        return ((((p-target)/30.)**2*mask).sum(1)[keep]/n[keep]).mean()
    p=m & (det!=0);outside=m & (det==0)
    anchor=sum(loss(pred[v],zero,m) for v in 'AB')/4
    anchor+=sum(.5*loss(pred[v],-det,p)+.5*loss(pred[v],zero,outside) for v in 'CD')/4
    total=anchor+.1*sum(smoothness(pred[v]/30.,batch['voiced']['A' if v in 'AB' else 'C'])[0] for v in 'ABCD')/4
    if condition=='bpdr' and response:
        scaled={v:p/30. for v,p in pred.items()}
        total+=boundary_invariance(scaled,m)[0]+counter_detuning(scaled,det/30.,m)[0]
    return total


def initialise(condition,norm):
    from bpdr_input.reporting.bpdr_showcase import load_model
    from bpdr_input.experiments.evaluation.transfer_study import frozen_runs
    base=dict(frozen_runs())[339]/(condition+'.pt')
    model,old,width=load_model(base)
    import torch
    # Exact linear reparameterization preserves the pretrained function despite
    # a new training-only normalizer. Output gain changes optimization, not units.
    with torch.no_grad():
        w=model.project.weight.detach().clone()
        model.project.weight.mul_(torch.from_numpy(norm.scale[:width]/old.scale[:width]))
        model.project.bias.add_(w@torch.from_numpy((norm.mean[:width]-old.mean[:width])/old.scale[:width]))
        model.head.weight.div_(30.);model.head.bias.div_(30.)
    saved=torch.load(base,weights_only=False,map_location='cpu')
    return model,width,saved


def evaluate(model,norm,width,records,device):
    import torch
    from bpdr_input.modeling.examples import collate
    from bpdr_input.experiments.learning.expanded_comparison import to_device
    rows=[];model.eval()
    with torch.inference_mode():
        for i,r in enumerate(records):
            p=pair(r);b=to_device(collate([p],norm,width),device)
            pred={v:(30.*model(x,b['padding']))[0].cpu().numpy() for v,x in b['inputs'].items()}
            rows.append(dict(score_predictions(p,pred),dataset=r['dataset'],group=r['singer_id'],phrase_id=r['phrase_id'],cents=r['cents'],corruption=r['corruption']))
    return rows


def passes(result):
    return all(s['median_response_gain']>=.25 and s['unshifted_alteration']<=1. and s['correction_gain']>0 for s in result['by_dataset'].values())


def train(mode='pilot'):
    import torch
    from bpdr_input.modeling.examples import Normalizer, collate
    from bpdr_input.experiments.learning.expanded_comparison import to_device
    from bpdr_input.experiments.learning.controlled_study import load_pair
    if not torch.xpu.is_available():raise RuntimeError('Declared XPU unavailable')
    torch.set_num_threads(2)
    base=Path(read(ROOT/'current.json')['path']);ev=read(base/'build_evidence.json')
    for split,h in ev['catalogs'].items():
        if sha(base/(split+'_catalog.json'))!=h:raise ValueError('Rebuilt catalog changed')
    train_records=read(base/'training_catalog.json')['records'];dev=read(base/'development_catalog.json')['records']
    if {(r['dataset'],r['singer_id']) for r in train_records}&{(r['dataset'],r['singer_id']) for r in dev}:raise ValueError('Group overlap')
    with np.load(base/'normalizer.npz') as z:norm=Normalizer(z['mean'],z['scale'])
    out=base/mode/uuid.uuid4().hex;out.mkdir(parents=True)
    if mode=='pilot':
        chosen=[]
        for ds in ['vocadito','jamendo']:
            by=defaultdict(list)
            for r in train_records:
                if r['dataset']==ds:by[r['phrase_id']].append(r)
            # First two phrases with both correction signs, no score-based selection.
            for key in sorted(by):
                rr=by[key];neg=sorted([r for r in rr if r['cents']<0],key=lambda r:(abs(r['cents']-(-30.)),r['corruption']))
                pos=sorted([r for r in rr if r['cents']>0],key=lambda r:(abs(r['cents']-30.),r['corruption']))
                if neg and pos:chosen.extend([neg[0],pos[0]])
                if sum(r['dataset']==ds for r in chosen)==4:break
        if len(chosen)!=8:raise ValueError('Need eight paired pilot records')
        conditions=['augmentation_only'];updates=200;schedule=[list(range(8))]*updates;synth=[]
    elif mode=='matched':
        pilot=read(Path(read(base/'pilot_current.json')['path'])/'evidence.json')
        if not pilot['training_gate_passes']:raise ValueError('Pilot learning gate failed; broad comparison stays locked')
        chosen=train_records;conditions=['augmentation_only','frame_only','bpdr'];updates=300
        data=Path(read(COMPONENT/'runs/controlled_study/current.json')['path'])
        synth=[r for r in read(data/'catalog.json')['records'] if r['split']=='train']
        rng=np.random.default_rng(339);weights=hierarchical_weights(chosen)
        schedule=[[*rng.choice(len(chosen),4,p=weights).tolist(),*(len(chosen)+rng.choice(len(synth),4,replace=False)).tolist()] for _ in range(updates)]
    else:raise ValueError('Unknown fit mode')
    declaration=dict(version='balanced-real-learning-2',mode=mode,source_sha256=sha(__file__),base=str(base),
        updates=updates,seed=339,conditions=conditions,output_gain=30.,optimizer='AdamW lr .001, decay .01, clip 1',
        anchor='Squared physical error / 900; C/D correction and preservation regions each half; A/B preserve all reliable frames',
        response='Boundary/detuning/smoothness on prediction and detuning /30; response starts after 100 updates',
        initialization='Synthetic seed339 corresponding condition, exact linear input normalization remap; divided head times30 during training',
        selected_records=[dict(dataset=r['dataset'],phrase_id=r['phrase_id'],cents=r['cents'],corruption=r['corruption']) for r in chosen],
        gate='Each dataset response >=.25, original alteration <=1 cent, positive correction gain; pilot gates further fitting on training only',
        development='Never for selection/normalization/early stopping; matched final endpoint only',
        final_test_access=False,research_ready_examples=0,review_status='pending')
    write(out/'declaration.json',declaration);np.save(out/'schedule.npy',np.asarray(schedule));(out/'executed_real_learning_v2.py').write_bytes(Path(__file__).read_bytes())
    results={};cache={};print('FIT',mode,out,flush=True)
    for condition in conditions:
        torch.manual_seed(339);model,width,saved=initialise(condition,norm);model.to('xpu').train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);history=[];start=time.monotonic()
        for step,indices in enumerate(schedule,1):
            model.train();optimizer.zero_grad();loss_value=0.
            for index in indices:
                index=int(index)
                if index not in cache:
                    if index<len(chosen):cache[index]=pair(chosen[index])
                    else:
                        p=load_pair(synth[index-len(chosen)]);p['inputs']={v:relative(x) for v,x in p['inputs'].items()};cache[index]=p
                b=to_device(collate([cache[index]],norm,width),'xpu')
                pred={v:30.*model(x,b['padding']) for v,x in b['inputs'].items()}
                loss=balanced_objective(pred,b,condition,step>100);(loss/len(indices)).backward();loss_value+=float(loss.detach())/len(indices)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();history.append(dict(update=step,loss=loss_value))
            if step%25==0:print(condition,step,'loss',round(loss_value,5),'seconds',round(time.monotonic()-start),flush=True)
        rows=evaluate(model,norm,width,chosen if mode=='pilot' else dev,'xpu');result=summaries(rows);results[condition]=result
        write(out/(condition+'_rows.json'),rows);write(out/(condition+'_history.json'),history)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};state['head.weight']*=30.;state['head.bias']*=30.
        saved.update(state=state,normalizer=norm.state(),input_transform='explicit_voicing_phrase_relative_fine_pitch_v2',
            data_status='exploratory_real_plus_synthetic' if mode=='matched' else 'training_only_overfit_diagnostic',declaration_sha256=sha(out/'declaration.json'))
        torch.save(saved,out/(condition+'.pt'));print('RESULT',condition,json.dumps(result['by_dataset']),flush=True)
        del model,optimizer
    evidence=dict(results=results,training_gate_passes=passes(results['augmentation_only']) if mode=='pilot' else None,
        development_gate_passes={c:passes(r) for c,r in results.items()} if mode=='matched' else None,
        declaration_sha256=sha(out/'declaration.json'),final_test_access=False,model_adopted=False)
    write(out/'evidence.json',evidence);write(base/(mode+'_current.json'),dict(path=str(out)));print('FIT_COMPLETE',out,flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='build':rebuild()
    else:train(sys.argv[1])
