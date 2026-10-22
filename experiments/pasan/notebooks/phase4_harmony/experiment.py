"""Train-only weights, fair ablations, locked validation selection, final test once."""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler

from .data import IGNORE,LABELS,digest,load_snapshot
from .model import FunctionTransformer,ModelConfig
from .metrics import metrics,selection_score,bar_oracle,subgroup_metrics,artist_bootstrap


@dataclass(frozen=True)
class RunConfig:
    epochs: int = 15
    patience: int = 5
    batch_size: int = 64
    learning_rate: float = .001
    weight_decay: float = .01
    seed: int = 42
    threads: int = 4


# Full 2 x 2 x 2 controlled factorial, plus two focused imbalance experiments
# and a local MLP baseline. Genre/weight ablations share initialization and shuffle seed.
def specifications():
    result=[{'name':'beat_local_mlp','resolution':'beat','genre':False,'weighted':False,'kind':'mlp','change_loss':0.,'balance_constants':False}]
    for resolution in ['beat','bar']:
        for genre in [False,True]:
            for weighted in [False,True]:
                result.append({'name':f'{resolution}_genre{int(genre)}_weight{int(weighted)}',
                               'resolution':resolution,'genre':genre,'weighted':weighted,'kind':'transformer',
                               'change_loss':0.,'balance_constants':False})
    result += [{'name':'beat_weighted_change','resolution':'beat','genre':False,'weighted':True,
                'kind':'transformer','change_loss':.3,'balance_constants':False},
               {'name':'beat_weighted_resample','resolution':'beat','genre':False,'weighted':True,
                'kind':'transformer','change_loss':0.,'balance_constants':True}]
    return result


def seed_all(seed,threads):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def class_weights(y,weighted):
    counts=np.bincount(y[y>=0],minlength=3)
    if (counts==0).any():raise ValueError(f'Training class absent: {counts}')
    weights=np.sqrt(counts.sum()/np.maximum(counts,1)) if weighted else np.ones(3)
    weights=weights/weights.mean()
    weights=np.clip(weights,.25,3.)
    return torch.tensor(weights,dtype=torch.float32),counts


def binary_weight(y):
    pairs=(y[:,:-1]>=0)&(y[:,1:]>=0)
    positive=int(((y[:,:-1]!=y[:,1:])&pairs).sum());negative=int(pairs.sum())-positive
    return min(negative/max(positive,1),10.),positive,negative


def dataset(arrays,indices,resolution):
    keys=['x','lengths','bar_index','mode','genres',f'y_{resolution}']
    return TensorDataset(*(torch.from_numpy(np.ascontiguousarray(arrays[k][indices])) for k in keys))


def loader(arrays,indices,resolution,config,shuffle=False,sample_weights=None):
    generator=torch.Generator().manual_seed(config.seed)
    sampler=WeightedRandomSampler(torch.tensor(sample_weights,dtype=torch.double),len(indices),replacement=True,generator=generator) if sample_weights is not None else None
    return DataLoader(dataset(arrays,indices,resolution),batch_size=config.batch_size,
                      shuffle=shuffle and sampler is None,sampler=sampler,generator=generator,num_workers=0)


def expand(pred,bar_index,resolution):
    if resolution=='beat':return pred
    return np.take_along_axis(pred,np.maximum(bar_index,0),axis=1)


@torch.no_grad()
def predict(model,arrays,indices,config,ablation=None):
    model.eval();all_pred=[];loss_sum=valid_count=0
    for x,lengths,bars,mode,genres,y in loader(arrays,indices,model.config.resolution,config):
        if ablation=='no_melody':x=x.clone();x[:,:,:72]=0
        if ablation=='no_genre':genres=torch.zeros_like(genres);genres[:,0]=1
        logits,_=model(x,lengths,bars,mode,genres)
        loss_sum+=float(F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=IGNORE,reduction='sum'))
        valid_count+=int((y>=0).sum())
        all_pred.append(logits.argmax(-1).numpy())
    native=np.concatenate(all_pred)
    pred=expand(native,arrays['bar_index'][indices],model.config.resolution)
    result=metrics(arrays['y_beat'][indices],pred)
    result['native_unweighted_ce']=loss_sum/max(valid_count,1)
    result['native_metrics']=metrics(arrays[f'y_{model.config.resolution}'][indices],native) if valid_count else None
    return pred,result


def atomic_json(path,obj):
    path=Path(path);temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False));temp.replace(path)


def atomic_checkpoint(path,obj):
    temp=Path(str(path)+'.partial');torch.save(obj,temp);temp.replace(path)


def run_one(arrays,examples,meta,snapshot,run_dir,spec,config):
    seed_all(config.seed,config.threads)
    model_config=ModelConfig(resolution=spec['resolution'],genre=spec['genre'],kind=spec['kind'])
    identity={'spec':spec,'run_config':asdict(config),'model_config':asdict(model_config),
              'dataset_metadata_sha256':digest(Path(snapshot)/'metadata.json'),
              'training_code_sha256':sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()}
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    if (run_dir/'result.json').exists():
        result=json.loads((run_dir/'result.json').read_text())
        if result['identity']!=identity:raise ValueError('Stale experiment: use a new run directory')
        return result
    train=np.flatnonzero(np.asarray(examples.split=='train'));val=np.flatnonzero(np.asarray(examples.split=='validation'))
    y=arrays[f'y_{spec["resolution"]}'][train]
    weights,counts=class_weights(y,spec['weighted']);pos_weight,positive,negative=binary_weight(y)
    sampling=None
    if spec['balance_constants']:
        # Keep constants in training; downweight their draw probability only.
        changing=np.asarray(examples.iloc[train].function_changes>0)
        sampling=np.where(changing,1.,.25)
    batches=loader(arrays,train,spec['resolution'],config,shuffle=True,sample_weights=sampling)
    model=FunctionTransformer(arrays['x'].shape[-1],len(meta['genre_vocab']),meta['config']['bars'],model_config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    best=-1.;stale=0;history=[]
    for epoch in range(1,config.epochs+1):
        start=time.monotonic();model.train();loss_sum=denominator=0.
        for x,lengths,bars,mode,genres,target in batches:
            if not (target>=0).any():continue
            optimizer.zero_grad(set_to_none=True)
            logits,change_logits=model(x,lengths,bars,mode,genres)
            raw=F.cross_entropy(logits.flatten(0,1),target.flatten(),weight=weights,ignore_index=IGNORE,reduction='sum')
            denom=weights[target[target>=0]].sum();loss=raw/denom
            if spec['change_loss']:
                valid=(target[:,:-1]>=0)&(target[:,1:]>=0)
                if valid.any():
                    changes=(target[:,:-1]!=target[:,1:]).float()
                    loss=loss+spec['change_loss']*F.binary_cross_entropy_with_logits(change_logits[valid],changes[valid],pos_weight=torch.tensor(pos_weight))
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            loss_sum+=float(raw.detach());denominator+=float(denom)
        pred,result=predict(model,arrays,val,config);score=selection_score(result)
        row={'epoch':epoch,'train_function_ce':loss_sum/max(denominator,1),'selection_score':score,
             'macro_f1':result['macro_f1'],'change_f1':result['change_f1'],'accuracy':result['accuracy'],
             'validation_native_ce':result['native_unweighted_ce'],'seconds':time.monotonic()-start}
        history.append(row)
        checkpoint={'identity':identity,'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                    'epoch':epoch,'history':history,'labels':LABELS,'genre_vocab':meta['genre_vocab'],
                    'bars':meta['config']['bars'],'feature_dim':meta['feature_dim'],'class_counts':counts.tolist(),
                    'class_weights':weights.tolist(),'change_pos_weight':pos_weight,'change_positive':positive,
                    'change_negative':negative,'validation':result,
                    'rng_state':torch.get_rng_state(),'shuffle_rng_state':batches.generator.get_state()}
        if score>best+1e-5:
            best=score;stale=0;checkpoint['best_score']=best
            atomic_checkpoint(run_dir/'best.pt',checkpoint)
        else:stale+=1
        checkpoint.update(best_score=best,stale_epochs=stale)
        atomic_checkpoint(run_dir/'latest.pt',checkpoint)
        pd.DataFrame(history).to_csv(run_dir/'history.csv',index=False)
        print(f'{spec["name"]} epoch {epoch:02d}: macro-F1={result["macro_f1"]:.3f}, change-F1={result["change_f1"]:.3f}, score={score:.3f}',flush=True)
        if stale>=config.patience:break
    checkpoint=torch.load(run_dir/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    pred,result=predict(model,arrays,val,config)
    _,no_melody=predict(model,arrays,val,config,ablation='no_melody')
    _,no_genre=predict(model,arrays,val,config,ablation='no_genre')
    result={'identity':identity,'best_epoch':checkpoint['epoch'],'validation':result,
            'validation_no_melody':no_melody,'validation_no_genre':no_genre,
            'parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'class_counts':counts.tolist(),'class_weights':weights.tolist(),'change_pos_weight':pos_weight,
            'subgroups':subgroup_metrics(arrays['y_beat'][val],pred,examples.iloc[val].reset_index(drop=True))}
    np.savez_compressed(run_dir/'validation_predictions.npz',indices=val,prediction=pred)
    atomic_json(run_dir/'result.json',result)
    return result


def baselines(arrays,examples):
    train=np.flatnonzero(np.asarray(examples.split=='train'));val=np.flatnonzero(np.asarray(examples.split=='validation'))
    y=arrays['y_beat'];counts=np.bincount(y[train][y[train]>=0],minlength=3);majority=int(counts.argmax())
    pred=np.full(y[val].shape,majority,np.int64)
    # Global training majority, tonic and per-mode training priors; no reference history.
    by_mode={int(mode):int(np.bincount(y[train[arrays['mode'][train]==mode]][y[train[arrays['mode'][train]==mode]]>=0],minlength=3).argmax()) for mode in [0,1]}
    mode_pred=np.array([np.full(y.shape[1],by_mode[int(mode)],np.int64) for mode in arrays['mode'][val]])
    return {'training_majority':metrics(y[val],pred),'always_tonic':metrics(y[val],np.zeros_like(pred)),
            'mode_majority':metrics(y[val],mode_pred),'bar_gold_accuracy_ceiling':bar_oracle(y[val],arrays['bar_index'][val],arrays['y_bar'].shape[1])}


def run_suite(snapshot,run_root,config=RunConfig(),specs=None):
    arrays,examples,meta=load_snapshot(snapshot);run_root=Path(run_root);run_root.mkdir(parents=True,exist_ok=True)
    specs=specifications() if specs is None else specs
    results=[run_one(arrays,examples,meta,snapshot,run_root/spec['name'],spec,config) for spec in specs]
    rows=[{'name':r['identity']['spec']['name'],'resolution':r['identity']['spec']['resolution'],
           'genre':r['identity']['spec']['genre'],'weighted':r['identity']['spec']['weighted'],
           'best_epoch':r['best_epoch'],**{k:r['validation'][k] for k in ['accuracy','macro_f1','change_f1','transition_pair_accuracy']},
           'selection_score':selection_score(r['validation']),
           'no_melody_macro_f1':r['validation_no_melody']['macro_f1']} for r in results]
    comparison=pd.DataFrame(rows).sort_values('selection_score',ascending=False)
    comparison.to_csv(run_root/'validation_comparison.csv',index=False)
    chosen=comparison.iloc[0]['name']
    selection={'name':chosen,'criterion':'.7 * validation common-beat macro-F1 + .3 * validation exact change-F1',
               'dataset_metadata_sha256':digest(Path(snapshot)/'metadata.json'),
               'best_checkpoint_sha256':digest(run_root/chosen/'best.pt'),
               'candidate_result_sha256':{s['name']:digest(run_root/s['name']/'result.json') for s in specs},
               'seed':config.seed,'test_used_for_selection':False}
    selection_path=run_root/'selection.json'
    if selection_path.exists() and read_json(selection_path)!=selection:
        raise ValueError('Frozen selection differs: use a new experiment directory')
    atomic_json(selection_path,selection)
    atomic_json(run_root/'baselines.json',baselines(arrays,examples))
    return comparison,selection


def read_json(path):return json.loads(Path(path).read_text())


def run_replicated_suite(snapshot,run_root,config=RunConfig(),seeds=(42,43,44),specs=None):
    """Choose a configuration by mean validation score, deploy the FIRST seed.

    Never choose the luckiest initialization or evaluate every seed on test.
    The order of seeds fixes the deployed checkpoint before observing results.
    """
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Distinct seeds required')
    run_root=Path(run_root);run_root.mkdir(parents=True,exist_ok=True)
    specs=specifications() if specs is None else specs
    frames=[]
    for seed in seeds:
        comparison,_=run_suite(snapshot,run_root/f'seed_{seed}',replace(config,seed=seed),specs)
        frames.append(comparison.assign(seed=seed))
    all_runs=pd.concat(frames,ignore_index=True)
    all_runs.to_csv(run_root/'validation_all_seeds.csv',index=False)
    columns=['accuracy','macro_f1','change_f1','transition_pair_accuracy','selection_score','no_melody_macro_f1']
    grouped=all_runs.groupby(['name','resolution','genre','weighted'],sort=False)
    comparison=grouped[columns].mean().reset_index()
    spread=grouped[['macro_f1','change_f1','selection_score']].std(ddof=1).add_suffix('_seed_sd').reset_index()
    comparison=comparison.merge(spread,on=['name','resolution','genre','weighted'],validate='one_to_one')
    comparison=comparison.sort_values('selection_score',ascending=False)
    comparison.to_csv(run_root/'validation_comparison.csv',index=False)
    chosen=str(comparison.iloc[0]['name']);deployed=f'seed_{seeds[0]}/{chosen}'
    selection={'name':deployed,'configuration':chosen,
               'criterion':'mean over fixed seeds of (.7 * validation common-beat macro-F1 + .3 * validation exact change-F1)',
               'seeds':list(seeds),'seed':seeds[0],'deployment_seed_rule':'first seed, fixed before training',
               'dataset_metadata_sha256':digest(Path(snapshot)/'metadata.json'),
               'best_checkpoint_sha256':digest(run_root/deployed/'best.pt'),
               'candidate_result_sha256':{f'seed_{seed}/{spec["name"]}':digest(run_root/f'seed_{seed}'/spec['name']/'result.json') for seed in seeds for spec in specs},
               'test_used_for_selection':False}
    path=run_root/'selection.json'
    if path.exists() and read_json(path)!=selection:raise ValueError('Frozen selection differs: use a new experiment directory')
    atomic_json(path,selection)
    atomic_json(run_root/'baselines.json',read_json(run_root/f'seed_{seeds[0]}'/'baselines.json'))
    return comparison,selection


def load_model(checkpoint_path,snapshot):
    arrays,examples,meta=load_snapshot(snapshot)
    ckpt=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if ckpt['identity']['dataset_metadata_sha256']!=digest(Path(snapshot)/'metadata.json'):
        raise ValueError('Checkpoint and snapshot mismatch')
    model=FunctionTransformer(ckpt['feature_dim'],len(ckpt['genre_vocab']),ckpt['bars'],ModelConfig(**ckpt['identity']['model_config']))
    model.load_state_dict(ckpt['state_dict']);model.eval()
    return model,arrays,examples,meta,ckpt


def final_test(snapshot,run_root):
    """Evaluate ONLY the frozen winner; repeated calls return the same saved result."""
    run_root=Path(run_root);selection=read_json(run_root/'selection.json');path=run_root/'test_result.json'
    checkpoint=run_root/selection['name']/'best.pt'
    if digest(checkpoint)!=selection['best_checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    if path.exists():
        previous=read_json(path)
        if previous['selection']!=selection:raise ValueError('Test already used under another selection')
        if digest(Path(snapshot)/'metadata.json')!=selection['dataset_metadata_sha256']:raise ValueError('Snapshot changed')
        return previous
    model,arrays,examples,meta,ckpt=load_model(checkpoint,snapshot)
    cfg=RunConfig(**ckpt['identity']['run_config']);seed_all(cfg.seed,cfg.threads)
    idx=np.flatnonzero(np.asarray(examples.split=='test'));train=np.flatnonzero(np.asarray(examples.split=='train'))
    pred,result=predict(model,arrays,idx,cfg)
    y=arrays['y_beat'][idx];counts=np.bincount(arrays['y_beat'][train][arrays['y_beat'][train]>=0],minlength=3)
    baseline=np.full(y.shape,int(counts.argmax()),np.int64)
    result={'selection':selection,'test':result,'training_majority':metrics(y,baseline),
            'paired_artist_bootstrap_95pct':artist_bootstrap(y,pred,baseline,examples.iloc[idx].reset_index(drop=True)),
            'subgroups':subgroup_metrics(y,pred,examples.iloc[idx].reset_index(drop=True)),
            'test_windows':len(idx),'test_tracks':int(examples.iloc[idx].track_id.nunique()),
            'test_artists':int(examples.iloc[idx].artist_id.nunique())}
    np.savez_compressed(run_root/'test_predictions.npz',indices=idx,prediction=pred)
    atomic_json(path,result)
    return result
