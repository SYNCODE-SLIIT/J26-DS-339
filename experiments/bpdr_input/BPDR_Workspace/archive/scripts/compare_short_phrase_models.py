"""Grouped engineering comparison on physical short deviations; never activates a model."""
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
import hashlib,json,sys,time
import joblib
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.ensemble import ExtraTreesRegressor

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
from probe_short_phrase_renderer import sha
from acquire_fresh_confirmation import check_lock
from bpdr_input.modeling import joint_transformer as old
from bpdr_input.modeling import short_phrase_transformer as short

DATA=WS/'results/transformer_usability/2026-11-02_v40_short_phrase_data'
ROOT=WS/'results/transformer_usability/2026-11-02_v41_short_phrase_models'
ARMS=['joint_retrained','short_transformer','short_extra_trees']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def norm(x):return json.loads(json.dumps(x,allow_nan=False))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def declare():
    check_lock();ROOT.mkdir(parents=True,exist_ok=True)
    receipt=read(DATA/'integrity_receipt.json');rows=read(DATA/'train_pairs.json');originals=read(DATA/'train_originals.json')
    assert receipt['all336_attempts_retained'] and receipt['training_singers_only']
    groups=sorted({s['group'] for s in originals})
    scoreable={r['group'] for r in rows if 'excluded' not in r}
    assert len(scoreable)>=4
    ordered=sorted(groups,key=lambda g:hashlib.sha256(('339:'+g).encode()).hexdigest())
    folds=[[] for _ in range(4)]
    for i,g in enumerate([g for g in ordered if g in scoreable]):folds[i%4].append(g)
    for i,g in enumerate([g for g in ordered if g not in scoreable]):folds[i%4].append(g)
    lock=read(DATA.parent/'2026-11-01_v37_fresh_confirmation/pipeline_lock.json')
    paths=[Path(__file__),Path(short.__file__),Path(old.__file__),
           WS/'source/bpdr_input/modeling/held_voice_transformer.py',WS/'source/bpdr_input/modeling/step_context.py']
    d=norm(dict(version='short-phrase-model-comparison-1',source_hashes={str(p.resolve()):sha(p) for p in paths},
        data_integrity_sha256=sha(DATA/'integrity_receipt.json'),input_hashes_sha256=sha(DATA/'input_hashes.json'),
        train_pairs_sha256=sha(DATA/'train_pairs.json'),train_originals_sha256=sha(DATA/'train_originals.json'),
        frozen_model=lock['model_checkpoint'],frozen_model_sha256=lock['model_checkpoint_sha256'],
        folds=folds,all_training_groups=groups,scoreable_groups=sorted(scoreable),
        arms=ARMS,seed=339,updates=800,batch_size=36,
        sampler='Twelve positive C plateau centers, twelve zero outside C centers, twelve original reliable centers per batch. Uniform eligible singer then record then frame; identical trace for all arms. Original-only singers remain in preservation sampling.',
        neural_loss='smooth_l1(prediction/30,target/30,beta=.2) +.1 BCE(activity_logit,target!=0); same loss for both Transformer variants; no reference supplied at inference',
        joint_config=asdict(old.JointConfig()),short_config=asdict(short.ShortConfig()),
        tree_config=dict(n_estimators=128,max_depth=18,min_samples_leaf=8,max_features=1.,random_state=339,n_jobs=2),
        tree_features='Same81x8 short context plus81 real padding flags, flattened. Identical sampled rows and targets; regression only, no neural activity loss; no equal wall-clock or optimal-tuning claim.',
        optimizer=dict(lr=.0005,weight_decay=.001),
        evaluation='Each fitted model predicts only held-out training singers. Correction errors on nonzero audited plateau; original preservation on every valid original; all336 attempts/21 sources/14 singers retained in denominators. No actual-WAV or natural-intention accuracy from raw predictions.',
        raw_gate='Every scoreable group: positive MAE<25c, signed response>=.25. Every represented original group: mean absolute raw prediction<=3c. Required before any qualification; not sufficient for activation.',
        no_development_or_other_holdout_access=True,research_validation_passed=False,production_eligible=False,
        limitations='Known training-source grouped development only; six scoreable singers may be a narrow subset. Frozen old model has a different training history. Retrained joint comparison changes representation as well as token density; only short Transformer vs short ExtraTrees shares identical input tokens.'))
    p=ROOT/'declaration.json'
    if p.exists():assert read(p)==d
    else:write(p,d)
    for p,h in d['source_hashes'].items():assert sha(p)==h
    assert sha(d['frozen_model'])==d['frozen_model_sha256']
    for p,h in read(DATA/'input_hashes.json').items():assert sha(p)==h
    return d,rows,originals


def records(rows,originals):
    result=[]
    for s in originals:
        if 'excluded' in s:continue
        with np.load(s['analysis_A']) as z:raw=z['raw_views'][0].copy();reliable=z['reliable'].copy()
        result.append(dict(id='A_'+s['source_id'],kind='original',group=s['group'],raw=raw,
            target=np.zeros(len(raw),np.float32),mask=reliable,source_id=s['source_id']))
    for index,r in enumerate(rows):
        if 'excluded' in r:continue
        with np.load(r['analysis_C']) as z:raw=z['raw_views'][0].copy()
        with np.load(r['target_path']) as z:target=z['target'].copy();mask=z['mask'].copy()
        result.append(dict(id='C_'+str(index),kind='corrupted',group=r['group'],raw=raw,target=target,
            mask=mask,source_id=r['source_id'],pair_index=index))
    for r in result:
        r['short_tokens'],r['short_pad']=short.tokens_from_features(short.frame_features(r['raw']),np.arange(len(r['raw'])))
        r['joint_tokens'],r['joint_pad']=old.joint_tokens(r['raw'],np.arange(len(r['raw'])))
    return result


def sample_trace(train,d):
    rng=np.random.default_rng(d['seed']);pools=[]
    for kind in ['positive','outside','original']:
        groups=defaultdict(list)
        for index,r in enumerate(train):
            mask=r['mask'] & ((r['target']!=0) if kind=='positive' else (r['target']==0))
            if (kind=='original')!=(r['kind']=='original') or not mask.any():continue
            groups[r['group']].append((index,np.flatnonzero(mask)))
        assert groups,'Missing sampler class '+kind
        pools.append(groups)
    trace=[]
    for _ in range(d['updates']):
        batch=[]
        for groups in pools:
            names=sorted(groups)
            for _ in range(12):
                entries=groups[names[int(rng.integers(len(names)))]]
                index,frames=entries[int(rng.integers(len(entries)))];center=int(rng.choice(frames))
                batch.append((index,center))
        trace.append(batch)
    trace=np.asarray(trace,dtype=np.int32)
    return trace,hashlib.sha256(trace.tobytes()).hexdigest()


def training_arrays(train,trace,prefix):
    flat=trace.reshape(-1,2)
    x=np.stack([train[i][prefix+'_tokens'][j] for i,j in flat])
    pad=np.stack([train[i][prefix+'_pad'][j] for i,j in flat])
    target=np.asarray([train[i]['target'][j] for i,j in flat],np.float32)
    return x,pad,target


def fit(train,arm,trace,d,folder):
    started=time.perf_counter();prefix='joint' if arm=='joint_retrained' else 'short'
    x,pad,y=training_arrays(train,trace,prefix)
    if arm=='short_extra_trees':
        model=ExtraTreesRegressor(**d['tree_config']);flat=np.concatenate([x.reshape(len(x),-1),pad],axis=1)
        model.fit(flat,y);joblib.dump(model,folder/'checkpoint.joblib')
        parameters=None
    else:
        torch.manual_seed(d['seed'])
        model=old.JointTransformer(old.JointConfig(**d['joint_config'])) if prefix=='joint' else short.ShortPhraseTransformer(short.ShortConfig(**d['short_config']))
        optimizer=torch.optim.AdamW(model.parameters(),**d['optimizer']);model.train();history=[]
        for step in range(d['updates']):
            a=step*d['batch_size'];b=a+d['batch_size']
            target=torch.from_numpy(y[a:b]);p,logit,_=model(torch.from_numpy(x[a:b]),torch.from_numpy(pad[a:b]),True)
            loss=F.smooth_l1_loss(p/30,target/30,beta=.2)+.1*F.binary_cross_entropy_with_logits(logit,(target!=0).float())
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if (step+1)%200==0:
                history.append(dict(update=step+1,loss=float(loss.detach())))
                print('SHORT_MODEL_FIT',folder.parent.name,arm,step+1,float(loss.detach()),flush=True)
        model.eval();parameters=sum(p.numel() for p in model.parameters())
        metadata=dict(training_groups=sorted({r['group'] for r in train}),training_sources=sorted({r['source_id'] for r in train}),
            declaration_sha256=sha(ROOT/'declaration.json'),updates=d['updates'],seed=d['seed'])
        (old.save if prefix=='joint' else short.save)(folder/'checkpoint.pt',model,metadata)
        write(folder/'loss_history.json',history)
    return model,dict(elapsed_seconds=time.perf_counter()-started,parameters=parameters,sampled_rows=len(y))


def predict(model,record,arm):
    prefix='joint' if arm in ['joint_retrained','frozen_joint'] else 'short'
    x,pad=record[prefix+'_tokens'],record[prefix+'_pad']
    if arm=='short_extra_trees':p=model.predict(np.concatenate([x.reshape(len(x),-1),pad],axis=1)).astype(np.float32)
    else:
        p=[];model.eval()
        with torch.inference_mode():
            for start in range(0,len(x),128):p.extend(model(torch.from_numpy(x[start:start+128]),torch.from_numpy(pad[start:start+128])).numpy())
        p=np.asarray(p,np.float32)
    p[record['raw'][:,66]<=.5]=0.
    assert np.isfinite(p).all() and np.max(abs(p))<=30.00001
    return p


def summarize(records,predictions):
    groups=[]
    for group in sorted({r['group'] for r in records}):
        positive=[];signed=[];original=[];zero=[]
        for r in records:
            if r['group']!=group:continue
            p=predictions[r['id']];m=r['mask'];t=r['target']
            if r['kind']=='original':
                if m.any():original.append(float(np.mean(abs(p[m]))))
            else:
                q=m&(t!=0);z=m&(t==0)
                if q.any():positive.append(float(np.mean(abs(p[q]-t[q]))));signed.append(float(np.mean(p[q]*t[q])/np.mean(t[q]**2)))
                if z.any():zero.append(float(np.mean(abs(p[z]))))
        groups.append(dict(group=group,positive_mae_cents=float(np.mean(positive)) if positive else None,
            signed_response=float(np.mean(signed)) if signed else None,
            original_mae_cents=float(np.mean(original)) if original else None,
            outside_mae_cents=float(np.mean(zero)) if zero else None,scoreable_pairs=len(positive)))
    positive=[g for g in groups if g['positive_mae_cents'] is not None]
    original=[g for g in groups if g['original_mae_cents'] is not None]
    return dict(groups=groups,positive_singer_macro_mae_cents=float(np.mean([g['positive_mae_cents'] for g in positive])),
        original_singer_macro_mae_cents=float(np.mean([g['original_mae_cents'] for g in original])),
        raw_gate_passed=all(g['positive_mae_cents']<25 and g['signed_response']>=.25 for g in positive) and all(g['original_mae_cents']<=3 for g in original))


def main():
    torch.set_num_threads(2);d,rows,originals=declare();data=records(rows,originals)
    frozen,_=old.load(d['frozen_model']);all_predictions={'frozen_joint':{r['id']:predict(frozen,r,'frozen_joint') for r in data}}
    for arm in ARMS:all_predictions[arm]={}
    for index,groups in enumerate(d['folds'],1):
        train=[r for r in data if r['group'] not in groups];test=[r for r in data if r['group'] in groups]
        assert {r['group'] for r in train}.isdisjoint(groups)
        trace,digest=sample_trace(train,d);fold=ROOT/f'fold_{index}';fold.mkdir(exist_ok=True)
        np.save(fold/'sampler_trace.npy',trace)
        for arm in ARMS:
            folder=fold/arm;folder.mkdir(exist_ok=True);receipt_path=folder/'fit_receipt.json'
            if receipt_path.exists():
                receipt=read(receipt_path);assert receipt['declaration_sha256']==sha(ROOT/'declaration.json') and receipt['sampler_sha256']==digest
                checkpoint=folder/receipt['checkpoint'];assert sha(checkpoint)==receipt['checkpoint_sha256']
                model=joblib.load(checkpoint) if arm=='short_extra_trees' else (old.load(checkpoint)[0] if arm=='joint_retrained' else short.load(checkpoint)[0])
            else:
                model,info=fit(train,arm,trace,d,folder)
                checkpoint=folder/('checkpoint.joblib' if arm=='short_extra_trees' else 'checkpoint.pt')
                receipt=dict(**info,checkpoint=checkpoint.name,checkpoint_sha256=sha(checkpoint),
                    declaration_sha256=sha(ROOT/'declaration.json'),sampler_sha256=digest,
                    training_groups=sorted({r['group'] for r in train}),heldout_groups=groups,
                    training_sources=sorted({r['source_id'] for r in train}),heldout_sources=sorted({r['source_id'] for r in test}))
                write(receipt_path,receipt)
            pred={r['id']:predict(model,r,arm) for r in test};all_predictions[arm].update(pred)
            np.savez_compressed(folder/'heldout_predictions.npz',**pred)
            print('SHORT_MODEL_FOLD_COMPLETE',index,arm,flush=True)
    summaries={arm:summarize(data,pred) for arm,pred in all_predictions.items()}
    for arm,pred in all_predictions.items():np.savez_compressed(ROOT/(arm+'_all_predictions.npz'),**pred)
    write(ROOT/'results.json',dict(declaration_sha256=sha(ROOT/'declaration.json'),arms=summaries,
        physical_attempts=336,scoreable_pairs=sum('excluded' not in r for r in rows),
        source_attempts=21,represented_original_sources=sum('excluded' not in s for s in originals),
        attempted_singers=14,scoreable_singers=len(d['scoreable_groups']),
        raw_predictions_only=True,delivered_audio_qualification=False,production_eligible=False))
    print('SHORT_MODEL_COMPARISON_COMPLETE',{a:(s['positive_singer_macro_mae_cents'],s['original_singer_macro_mae_cents'],s['raw_gate_passed']) for a,s in summaries.items()},flush=True)


if __name__=='__main__':main()
