"""Declared gated-Transformer fitting and grouped evaluation; no demo activation."""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from bpdr_input.experiments.learning import context_improvement as existing
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.modeling.held_voice_transformer import HeldConfig,HeldVoiceTransformer,tokens,predict_raw,save,load
from bpdr_input.modeling.correction_accuracy import prediction_row,summarize,assert_independent
from bpdr_input.repair.held_voice_policy import HeldPolicy,proposal
from bpdr_input.reporting.transformer_accuracy_report import complete_report

ROOT=existing.WORKSPACE/'results/transformer_usability/2026-10-28_v1'


def prepared(rows):
    pairs=recovery.cached_pairs(rows)
    for p in pairs:
        if 'excluded' not in p:
            assert p['audit']['passed']
            assert p['mask'].shape==p['target'].shape
    return pairs


def anchor_pool(sources,exclude_groups=()):
    result=[];cache={}
    for source in sources:
        group=source['source']['singer_id']
        if group in exclude_groups or 'working_audio' not in source:continue
        path=Path(source['working_audio'])
        _,analysis=recovery.from_cached(str(path),path.parent/'original_features.npz',cache)
        if not analysis['reliable'].any():continue
        result.append(dict(group=group,path_A=str(path),A=analysis,source=source['source']))
    return result


def fit(pairs,arm,seed,updates,folder,declaration_hash,config=None,anchors=None):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    model=HeldVoiceTransformer(config or HeldConfig(gated=arm=='gated'))
    assert model.config.gated==(arm=='gated')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    valid=[p for p in pairs if 'excluded' not in p]
    grouped=defaultdict(list)
    for p in valid: grouped[p['group']].append(p)
    groups=sorted(grouped)
    originals=anchors if anchors is not None else list({p['path_A']:p for p in valid}.values())
    if not originals:raise ValueError('No reliable original anchors available')
    original_groups={g:[p for p in originals if p['group']==g] for g in sorted({p['group'] for p in originals})}
    original_group_names=sorted(original_groups)
    cache={}
    # Cache full input windows once; labels never enter this feature transform.
    for p in valid:
        for view in ('A','C'):
            key=p['path_'+view]
            if key not in cache:
                cache[key]=[tokens(raw,np.arange(len(raw)),model.config) for raw in p[view]['views']]
    for p in originals:
        if p['path_A'] not in cache:
            cache[p['path_A']]=[tokens(raw,np.arange(len(raw)),model.config) for raw in p['A']['views']]
    history=[];started=time.perf_counter();model.train()
    folder.mkdir(parents=True,exist_ok=True)
    for step in range(updates):
        xx=[];pp=[];targets=[];weights=[]
        for _ in range(8):
            g=groups[int(rng.integers(len(groups)))];p=grouped[g][int(rng.integers(len(grouped[g])))]
            plateau=p['mask']&(p['target']!=0);outside=p['mask']&(p['target']==0)
            index=int(rng.choice(np.flatnonzero(plateau if rng.random()<.7 or not outside.any() else outside)))
            alternate=int(rng.integers(1,3))
            for view,boundary in (('A',0),('A',alternate),('C',0),('C',alternate)):
                x,pad=cache[p['path_'+view]][boundary];xx.append(x[index]);pp.append(pad[index])
                target=0. if view=='A' else float(p['target'][index])/30.
                targets.append(target);weights.append(4. if view=='A' else 2. if target==0 else 1.)
            og=original_group_names[int(rng.integers(len(original_group_names)))] if anchors is not None else g
            op=original_groups[og][int(rng.integers(len(original_groups[og])))]
            oi=int(rng.choice(np.flatnonzero(op['A']['reliable'])))
            x,pad=cache[op['path_A']][0];xx.append(x[oi]);pp.append(pad[oi]);targets.append(0.);weights.append(4.)
        target=torch.tensor(targets);weight=torch.tensor(weights)
        output,logit,_=model(torch.from_numpy(np.stack(xx)),torch.from_numpy(np.stack(pp)),True)
        prediction=output/30.
        anchor=((prediction-target).square()*weight).mean()
        paired=prediction.reshape(8,5)[:,:4];wanted=target.reshape(8,5)[:,2]
        response=.5*((paired[:,2]-paired[:,0]-wanted).square().mean()+(paired[:,3]-paired[:,1]-wanted).square().mean())
        boundary=.5*((paired[:,0]-paired[:,1]).square().mean()+(paired[:,2]-paired[:,3]).square().mean())
        activity=F.binary_cross_entropy_with_logits(logit,(target!=0).float())
        loss=anchor+.25*response+.1*boundary+(.25*activity if model.config.gated else 0.)
        if not torch.isfinite(loss): raise ValueError('Non-finite loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%100==0 or step==updates-1:
            history.append(dict(update=step+1,loss=float(loss.detach()),anchor=float(anchor.detach()),activity=float(activity.detach())))
            existing.write(folder/'training_progress.json',dict(update=step+1,planned_updates=updates,history=history))
    folder.mkdir(parents=True,exist_ok=True)
    metadata=dict(arm=arm,seed=seed,updates=updates,training_groups=sorted(set(groups)|set(original_groups)),training_sources=sorted({p['source']['sha256'] for p in valid+originals}),original_anchor_recordings=len(originals),declaration_sha256=declaration_hash,eligible_for_demo=False,research_validation_passed=False)
    save(folder/'checkpoint.pt',model.eval(),metadata)
    restored,payload=load(folder/'checkpoint.pt')
    probe=valid[0]['A']['views'][0]
    np.testing.assert_array_equal(predict_raw(model,probe)[0],predict_raw(restored,probe)[0])
    receipt=dict(**metadata,checkpoint_sha256=existing.sha(folder/'checkpoint.pt'),fit_seconds=time.perf_counter()-started,reload_verified=True,parameters=sum(p.numel() for p in model.parameters()),history=history)
    existing.write(folder/'fit_receipt.json',receipt)
    return restored,payload,receipt


def predictions(model,pair,view):
    return np.stack([predict_raw(model,raw)[0] for raw in pair[view]['views']])


def evaluate(model,payload,pairs,folder,policy=HeldPolicy(),render=False):
    # The pre-existing evaluator supplies unchanged renderer/encoded audits.
    # Only model prediction and the explicitly versioned proposal are adapted.
    original_prediction,original_proposal=existing.predictions,existing.legacy_proposal
    def propose(p,view,pred):
        detail=proposal(p['audio_'+view],p[view]['features']['times_seconds'],p[view]['views'],pred,policy)
        return detail['curve'],detail['plan'],detail['allowed']
    existing.predictions=predictions;existing.legacy_proposal=propose
    try: result=existing.evaluate(model,payload,pairs,folder,render=render,seed=payload['seed'])
    finally: existing.predictions=original_prediction;existing.legacy_proposal=original_proposal
    result.update(eligibility_policy=policy.__dict__,model_version='held-voice-transformer-1')
    existing.write(folder/'accuracy.json',result)
    existing.write_report(folder,result,'Held-voice Transformer accuracy')
    complete_report(folder)
    return result


def cv_worker(task):
    if len(task)==3: root,fold,arm=task;root=Path(root)
    else: fold,arm=task;root=ROOT
    declaration=existing.read(root/'declaration.json');folder=root/'cross_validation'/f'fold_{fold+1}'/arm
    if (folder/'validation/accuracy.json').exists():return existing.read(folder/'validation/accuracy.json')
    rows=existing.read(root/'train_pairs.json');held=set(declaration['folds'][fold])
    train=prepared([p for p in rows if p.get('group',p['source']['singer_id']) not in held])
    test_rows=[p for p in rows if p.get('group',p['source']['singer_id']) in held]
    if (folder/'fit_receipt.json').exists():
        receipt=existing.read(folder/'fit_receipt.json');assert receipt['checkpoint_sha256']==existing.sha(folder/'checkpoint.pt')
        model,payload=load(folder/'checkpoint.pt')
    else:
        anchors=anchor_pool(existing.read(root/'anchor_sources.json'),held) if (root/'anchor_sources.json').exists() else None
        model,payload,_=fit(train,arm,339,declaration.get('cv_updates',800),folder,existing.sha(root/'declaration.json'),
            HeldConfig(gated=arm=='gated',**declaration.get('model_config',{})),anchors)
    assert_independent(payload['training_groups'],payload['training_sources'],test_rows)
    return evaluate(model,payload,prepared(test_rows),folder/'validation')
