"""Matched paired-response learning, with/without virtual F0 pretraining; no promotion."""
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
import sys,time,hashlib
import numpy as np
import torch
from torch.nn import functional as F

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
import compare_short_phrase_models as previous
from probe_short_phrase_renderer import sha
from bpdr_input.modeling import short_phrase_transformer as model_module
from bpdr_input.experiments.learning import virtual_short_examples as virtual
from bpdr_input.repair.rule_correction import segment_notes,RulePolicy
from acquire_fresh_confirmation import check_lock

DATA=WS/'results/transformer_usability/2026-11-02_v45_native_short_pairs'
ROOT=WS/'results/transformer_usability/2026-11-02_v46_paired_short_models'
ARMS=['paired_physical','virtual_then_paired']
read,write,norm=previous.read,previous.write,previous.norm


def declare():
    check_lock();ROOT.mkdir(exist_ok=True);receipt=read(DATA/'integrity_receipt.json')
    rows=read(DATA/'train_pairs.json');originals=read(DATA/'train_originals.json')
    assert receipt['staged_pairs_verified']==115 and receipt['all_attempts_retained']==336
    for p,h in read(DATA/'input_hashes.json').items():assert sha(p)==h,p
    paths=[Path(__file__),Path(model_module.__file__),Path(virtual.__file__),WS/'tests/test_virtual_short_examples.py']
    old=read(previous.ROOT/'declaration.json')
    d=norm(dict(version='paired-short-response-comparison-1',source_hashes={str(p.resolve()):sha(p) for p in paths},
        data_integrity_sha256=sha(DATA/'integrity_receipt.json'),input_hashes_sha256=sha(DATA/'input_hashes.json'),
        folds=old['folds'],seed=339,updates=1200,batch_size=36,forward_rows_per_update=48,
        arms=ARMS,config=asdict(model_module.ShortConfig()),virtual_updates=400,physical_tail_updates=800,
        sampler='Reuse12-positive/12-known-zero/12-original singer-balanced physical trace; also infer the positive frame on its paired original A. Both arms use identical physical tail800 and same total1200 updates.',
        loss='0.5*(pairedA^2+(C-target)^2)+0.25*(C-pairedA-target)^2+0.5*Czero^2+0.5*generalA^2+0.05*BCE(activity,target!=0), corrections normalized30c',
        virtual='First400 updates in augmented arm only: source-training original context plus shared deliberate4-7Hz vibrato0-20c and quadratic drift+-15c; add randomized+-20/30c local160ms-smoothed pulses of registered widths. Same original source and expression in clean/corrupted pair. Confidence/voicing/energy unchanged. These are feature counterfactuals, not independently observed audio or extra singers.',
        virtual_widths_seconds=[.24,.28,.36,.50],source_context_seconds=.15,
        virtual_expression_probability=.5,virtual_pulse_kernel=[.1,.2,.4,.2,.1],
        virtual_zero_sampling='Half prefer near-event zero frames within200ms, remainder any unchanged voiced frame.',
        optimizer=dict(lr=.0005,weight_decay=.001),same_initial_seed=True,
        gate='Unchanged all-group raw gate: positive MAE<25c,response>=.25; every represented original group<=3c. Actual-WAV and known-zero preservation required separately.',
        scope='Training-source grouped engineering study only. Not proposal BPDR-vs-augmentation-only/frame-only research validation; boundary invariance remains by input construction.',
        nontraining_sources_accessed=False,research_validation_passed=False,production_eligible=False))
    path=ROOT/'declaration.json'
    if path.exists():assert read(path)==d
    else:write(path,d)
    for p,h in d['source_hashes'].items():assert sha(p)==h,p
    return d,rows,originals


def records(rows,originals):
    result=[]
    for source in originals:
        if 'excluded' in source:continue
        with np.load(source['analysis_A']) as z:raw=z['raw_views'][0].copy();mask=z['reliable'].copy()
        result.append(dict(id='A_'+source['source_id'],source_id=source['source_id'],group=source['group'],kind='original',raw=raw,target=np.zeros(len(raw),np.float32),mask=mask))
    for index,row in enumerate(rows):
        if 'excluded' in row:continue
        with np.load(row['analysis_C']) as z:raw=z['raw_views'][0].copy()
        with np.load(row['target_path']) as z:target=z['target'].copy();mask=z['mask'].copy()
        result.append(dict(id='C_'+str(index),source_id=row['source_id'],group=row['group'],kind='corrupted',raw=raw,target=target,mask=mask,pair_index=index))
    for r in result:
        r['short_tokens'],r['short_pad']=model_module.tokens_from_features(model_module.frame_features(r['raw']),np.arange(len(r['raw'])))
    return result


def virtual_pools(train,d):
    groups=defaultdict(list)
    for i,r in enumerate(train):
        if r['kind']!='original':continue
        regions=segment_notes(r['raw'][:,64],r['raw'][:,66]>.5,RulePolicy(note_split_cents=70.))
        options=[(a,b,w) for a,b in regions for w in d['virtual_widths_seconds'] if (b-a)*.02>=w+.30-1e-12]
        if options:groups[r['group']].append((i,options))
    assert groups
    return groups


def draw_virtual(train,pools,rng,d):
    names=sorted(pools);entries=pools[names[int(rng.integers(len(names)))]]
    index,options=entries[int(rng.integers(len(entries)))];a,b,width=options[int(rng.integers(len(options)))]
    low=a*.02+.15;high=b*.02-.15-width
    left=float(rng.uniform(low,max(low,high)))
    cents=float(rng.choice([-30.,-20.,20.,30.]));express=bool(rng.random()<.5)
    spec=dict(left_seconds=left,width_seconds=width,cents=cents,
        vibrato_amplitude_cents=float(rng.uniform(0,20)) if express else 0.,
        vibrato_hz=float(rng.uniform(4,7)),phase=float(rng.uniform(0,2*np.pi)),
        drift_cents=float(rng.uniform(-15,15)) if express else 0.)
    source=train[index];ar,cr,labels=virtual.make_pair(source['raw'],**spec)
    pos=int(rng.choice(np.flatnonzero(labels['positive'])))
    times=np.arange(len(cr))*.02;near=labels['zero']&(times>=left-.2)&(times<=left+width+.2)
    zero=near if rng.random()<.5 and near.any() else labels['zero']
    outside=int(rng.choice(np.flatnonzero(zero)))
    ax,ap=model_module.tokens_from_features(model_module.frame_features(ar),[pos])
    cx,cp=model_module.tokens_from_features(model_module.frame_features(cr),[pos,outside])
    trace=dict(source_id=source['source_id'],group=source['group'],positive_frame=pos,zero_frame=outside,**spec)
    return (ax[0],ap[0]),(cx[0],cp[0]),(cx[1],cp[1]),float(labels['target'][pos]),trace


def fit(train,arm,trace,d,fold,folder):
    torch.manual_seed(d['seed']);model=model_module.ShortPhraseTransformer(model_module.ShortConfig(**d['config']))
    optimizer=torch.optim.AdamW(model.parameters(),**d['optimizer']);originals={r['source_id']:r for r in train if r['kind']=='original'}
    rng=np.random.default_rng(d['seed']+7919+fold);pools=virtual_pools(train,d);virtual_trace=[];history=[]
    start=time.perf_counter();model.train()
    for step,batch in enumerate(trace):
        arows=[];crows=[];zeros=[];original_rows=[];targets=[]
        if arm=='virtual_then_paired' and step<d['virtual_updates']:
            for _ in range(12):
                a,c,z,target,record=draw_virtual(train,pools,rng,d)
                arows.append(a);crows.append(c);zeros.append(z);targets.append(target)
                virtual_trace.append(dict(update=step,**record))
        else:
            for i,j in batch[:12]:
                c=train[i];a=originals[c['source_id']]
                arows.append((a['short_tokens'][j],a['short_pad'][j]));crows.append((c['short_tokens'][j],c['short_pad'][j]));targets.append(float(c['target'][j]))
            for i,j in batch[12:24]:zeros.append((train[i]['short_tokens'][j],train[i]['short_pad'][j]))
        for i,j in batch[24:]:original_rows.append((train[i]['short_tokens'][j],train[i]['short_pad'][j]))
        packed=arows+crows+zeros+original_rows
        x=torch.from_numpy(np.stack([p[0] for p in packed]));pad=torch.from_numpy(np.stack([p[1] for p in packed]))
        target=torch.tensor(targets,dtype=torch.float32)/30
        pred,logit,_=model(x,pad,True);p=pred/30
        anchor=.5*(p[:12].square().mean()+(p[12:24]-target).square().mean())
        response=.25*(p[12:24]-p[:12]-target).square().mean()
        preservation=.5*p[24:36].square().mean()+.5*p[36:].square().mean()
        active=torch.cat([torch.zeros(12),torch.ones(12),torch.zeros(24)])
        loss=anchor+response+preservation+.05*F.binary_cross_entropy_with_logits(logit,active)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (step+1)%200==0:
            history.append(dict(update=step+1,loss=float(loss.detach())))
            print('PAIRED_SHORT_FIT',fold,arm,step+1,float(loss.detach()),flush=True)
    metadata=dict(arm=arm,training_groups=sorted({r['group'] for r in train}),training_sources=sorted({r['source_id'] for r in train}),
        declaration_sha256=sha(ROOT/'declaration.json'),updates=d['updates'],seed=d['seed'])
    model.eval();model_module.save(folder/'checkpoint.pt',model,metadata)
    write(folder/'virtual_trace.json',virtual_trace);write(folder/'loss_history.json',history)
    return model,dict(elapsed_seconds=time.perf_counter()-start,virtual_examples=len(virtual_trace),
        virtual_trace_sha256=sha(folder/'virtual_trace.json'),parameters=sum(p.numel() for p in model.parameters()))


def main():
    torch.set_num_threads(2);d,rows,originals=declare();data=records(rows,originals);all_predictions={a:{} for a in ARMS}
    for fold,groups in enumerate(d['folds'],1):
        train=[r for r in data if r['group'] not in groups];test=[r for r in data if r['group'] in groups]
        trace,digest=previous.sample_trace(train,d);base=ROOT/f'fold_{fold}';base.mkdir(exist_ok=True)
        np.save(base/'physical_trace.npy',trace)
        for arm in ARMS:
            folder=base/arm;folder.mkdir(exist_ok=True);path=folder/'fit_receipt.json'
            if path.exists():
                receipt=read(path);assert receipt['declaration_sha256']==sha(ROOT/'declaration.json') and receipt['physical_trace_sha256']==digest
                assert receipt['checkpoint_sha256']==sha(folder/'checkpoint.pt')
                model,_=model_module.load(folder/'checkpoint.pt')
            else:
                model,info=fit(train,arm,trace,d,fold,folder)
                receipt=dict(**info,declaration_sha256=sha(ROOT/'declaration.json'),physical_trace_sha256=digest,
                    checkpoint_sha256=sha(folder/'checkpoint.pt'),training_groups=sorted({r['group'] for r in train}),heldout_groups=groups,
                    training_sources=sorted({r['source_id'] for r in train}),heldout_sources=sorted({r['source_id'] for r in test}))
                write(path,receipt)
            pred={r['id']:previous.predict(model,r,'short_transformer') for r in test}
            np.savez_compressed(folder/'heldout_predictions.npz',**pred);all_predictions[arm].update(pred)
            print('PAIRED_SHORT_FOLD_COMPLETE',fold,arm,flush=True)
    summaries={arm:previous.summarize(data,pred) for arm,pred in all_predictions.items()}
    for arm,pred in all_predictions.items():np.savez_compressed(ROOT/(arm+'_all_predictions.npz'),**pred)
    write(ROOT/'results.json',dict(declaration_sha256=sha(ROOT/'declaration.json'),arms=summaries,
        physical_attempts=336,scoreable_pairs=115,original_attempts=21,original_controls=19,
        raw_predictions_only=True,delivered_audio_qualification=False,production_eligible=False))
    print('PAIRED_SHORT_MODELS_COMPLETE',{a:(v['positive_singer_macro_mae_cents'],v['original_singer_macro_mae_cents'],v['raw_gate_passed']) for a,v in summaries.items()},flush=True)


if __name__=='__main__':main()
