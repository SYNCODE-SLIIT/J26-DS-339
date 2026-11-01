"""Learned calibration of target-free acoustic step features; real-WAV labels."""
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from bpdr_input.experiments.learning import held_voice_training as held
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.step_context import evidence
from bpdr_input.modeling.step_transformer import StepTransformer,StepConfig,windows,predict_raw,save,load
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.repair.held_voice_policy import HeldPolicy
from bpdr_input.reporting.transformer_accuracy_report import complete_report

prepared=held.prepared
anchor_pool=held.anchor_pool


def predictions(model,pair,view):
    one=predict_raw(model,pair[view]['views'][0])[0]
    return np.stack([one]*3)


def evaluate(model,payload,pairs,folder,policy=None,render=False):
    policy=policy or HeldPolicy(version='held-voice-input-policy-2',region_split_cents=70.)
    previous=held.predictions;held.predictions=predictions
    try:result=held.evaluate(model,payload,pairs,folder,policy,render)
    finally:held.predictions=previous
    result.update(model_version='step-context-transformer-1',feature_contract='input-only-step-context-2',boundary_invariance='By engineered input construction; not evidence of learned BPDR-objective advantage')
    ex.write(folder/'accuracy.json',result);ex.write_report(folder,result,'Step-conditioned Transformer accuracy');complete_report(folder)
    return result


def fit(pairs,anchors,seed,updates,folder,declaration_hash):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    rng=np.random.default_rng(seed);model=StepTransformer();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    valid=[p for p in pairs if 'excluded' not in p];grouped=defaultdict(list)
    for p in valid:grouped[p['group']].append(p)
    groups=sorted(grouped);originals=defaultdict(list)
    for p in anchors:originals[p['group']].append(p)
    original_groups=sorted(originals)
    if not groups or not original_groups:raise ValueError('No supervised step or reliable original anchors')
    cache={}
    for p in valid:
        for view in ('A','C'):
            path=p['path_'+view]
            if path not in cache:cache[path]=windows(evidence(p[view]['views'][0]),np.arange(len(p[view]['views'][0])),model.config)
    for p in anchors:
        if p['path_A'] not in cache:cache[p['path_A']]=windows(evidence(p['A']['views'][0]),np.arange(len(p['A']['views'][0])),model.config)
    folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter();history=[];model.train()
    for step in range(updates):
        xx=[];padding=[];targets=[];weights=[]
        for _ in range(12):
            group=groups[int(rng.integers(len(groups)))];p=grouped[group][int(rng.integers(len(grouped[group])))]
            plateau=p['mask']&(p['target']!=0);outside=p['mask']&(p['target']==0)
            index=int(rng.choice(np.flatnonzero(plateau if rng.random()<.7 or not outside.any() else outside)))
            for view in ('A','C'):
                x,pad=cache[p['path_'+view]];xx.append(x[index]);padding.append(pad[index])
                target=0. if view=='A' else float(p['target'][index])/30.;targets.append(target);weights.append(4. if view=='A' else 2. if target==0 else 1.)
            og=original_groups[int(rng.integers(len(original_groups)))];op=originals[og][int(rng.integers(len(originals[og])))]
            oi=int(rng.choice(np.flatnonzero(op['A']['reliable'])));x,pad=cache[op['path_A']]
            xx.append(x[oi]);padding.append(pad[oi]);targets.append(0.);weights.append(4.)
        target=torch.tensor(targets);weight=torch.tensor(weights)
        output,logit,amplitude=model(torch.from_numpy(np.stack(xx)),torch.from_numpy(np.stack(padding)),True)
        pred=output/30.;positive=target!=0
        anchor=((pred-target).square()*weight).mean()
        paired=pred.reshape(12,3);desired=target.reshape(12,3)[:,1]
        response=(paired[:,1]-paired[:,0]-desired).square().mean()
        activity=F.binary_cross_entropy_with_logits(logit,positive.float())
        amp=((amplitude[positive]/30.-target[positive]).square().mean() if positive.any() else amplitude.sum()*0.)
        loss=anchor+.25*response+.5*activity+.25*amp
        if not torch.isfinite(loss):raise ValueError('Nonfinite step-Transformer loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%100==0 or step==updates-1:
            history.append(dict(update=step+1,loss=float(loss.detach()),anchor=float(anchor.detach()),activity=float(activity.detach())))
            ex.write(folder/'training_progress.json',dict(update=step+1,planned_updates=updates,history=history))
    metadata=dict(arm='step_conditioned',seed=seed,updates=updates,training_groups=sorted(set(groups)|set(originals)),training_sources=sorted({p['source']['sha256'] for p in valid+anchors}),original_anchor_recordings=len(anchors),declaration_sha256=declaration_hash,eligible_for_demo=False,research_validation_passed=False)
    save(folder/'checkpoint.pt',model.eval(),metadata);restored,payload=load(folder/'checkpoint.pt')
    probe=valid[0]['A']['views'][0];np.testing.assert_allclose(predict_raw(model,probe)[0],predict_raw(restored,probe)[0],atol=1e-6)
    receipt=dict(**metadata,checkpoint_sha256=ex.sha(folder/'checkpoint.pt'),fit_seconds=time.perf_counter()-started,reload_verified=True,parameters=sum(p.numel() for p in model.parameters()),history=history)
    ex.write(folder/'fit_receipt.json',receipt);return restored,payload,receipt


def cv_worker(task):
    root,fold=task;root=Path(root);torch.set_num_threads(2)
    declaration=ex.read(root/'declaration.json');groups=set(declaration['folds'][fold]);rows=ex.read(root/'train_pairs.json')
    train=[p for p in rows if p.get('group',p['source']['singer_id']) not in groups];test=[p for p in rows if p.get('group',p['source']['singer_id']) in groups]
    folder=root/f'cross_validation/fold_{fold+1}/step_conditioned'
    if (folder/'validation/accuracy.json').exists():return ex.read(folder/'validation/accuracy.json')
    if (folder/'fit_receipt.json').exists():
        receipt=ex.read(folder/'fit_receipt.json');assert ex.sha(folder/'checkpoint.pt')==receipt['checkpoint_sha256'];model,payload=load(folder/'checkpoint.pt')
    else:model,payload,_=fit(prepared(train),anchor_pool(ex.read(root/'anchor_sources.json'),groups),339,1000,folder,ex.sha(root/'declaration.json'))
    assert_independent(payload['training_groups'],payload['training_sources'],test)
    return evaluate(model,payload,prepared(test),folder/'validation')
